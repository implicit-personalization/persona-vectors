"""
Generate persona steering vectors from pre-extracted activations.

Two contrastive approaches
──────────────────────────────────────────────────────────────────────────────
For each persona:

  negative prompt  ->  Templated prompt + Question + Answer
  positive prompt  ->  Biography + Question + Answer

Load the saved mean response-token hidden states at STEER_LAYER. The activation
artifacts are already averaged across QA pairs and masked tokens, so each one
is a single (num_layers, hidden_size) tensor per (variant, persona).

  steering_vector = biography_h[layer] - templated_h[layer]

   Removes the shared "biography mode" direction and isolates what makes each
   persona's representation distinct from the average.

method="mean":  steering_vector = mean(d_i)
method="pca":   steering_vector = first right singular vector of centered
                diff matrix D, sign-aligned with mean(d_i) and scaled to
                match its L2 norm.

Centering (center=True):
  Before computing diffs, each per-question hidden vector is centered by
  subtracting its own feature-dimension mean, removing the per-vector DC
  offset that is common across all token positions and prompt variants.
"""

import json
from pathlib import Path
from typing import Literal

import torch
from rich.console import Console
from rich.table import Table
from safetensors.torch import load_file, save_file

from persona_vectors.artifacts import ActivationStore

STEER_LAYER = 20

console = Console()


def _compute_layer_sv(
    pos_stack: torch.Tensor,
    neg_stack: torch.Tensor,
    method: Literal["mean", "pca"],
    center: bool,
) -> tuple[torch.Tensor, float, float]:
    """Compute a steering vector for one layer's activations.

    Returns (raw_sv, sv_norm, suggested_alpha).
    """
    if center:
        pos_stack = pos_stack - pos_stack.mean(dim=-1, keepdim=True)
        neg_stack = neg_stack - neg_stack.mean(dim=-1, keepdim=True)

    diffs = pos_stack - neg_stack  # (n_questions, hidden_dim)

    if method == "mean":
        raw_sv = diffs.mean(dim=0)
    elif method == "pca":
        D_c = diffs - diffs.mean(dim=0)
        _, _, Vt = torch.linalg.svd(D_c, full_matrices=False)
        direction = Vt[0]
        mean_diff = diffs.mean(dim=0)
        if direction.dot(mean_diff) < 0:
            direction = -direction
        raw_sv = direction * mean_diff.norm()
    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'mean' or 'pca'.")

    sv_norm = raw_sv.norm().item()
    mean_rms = neg_stack.pow(2).mean(dim=-1).sqrt().mean().item()
    suggested_alpha = (20.0 * mean_rms) / (sv_norm + 1e-8)
    return raw_sv, sv_norm, suggested_alpha


def _load_pos_neg(
    store: ActivationStore,
    persona_id: str,
    verbose: bool,
) -> tuple[torch.Tensor, torch.Tensor, list[str]] | None:
    """Load and validate biography/templated activation pairs."""
    try:
        pos_activations = store.load("biography", persona_id)
    except FileNotFoundError:
        if verbose:
            print("✗ Biography activations not found. Run extraction first.")
        return None

    try:
        neg_activations = store.load("templated", persona_id)
    except FileNotFoundError:
        if verbose:
            print("✗ Templated activations not found. Run extraction first.")
        return None

    return pos_activations.float(), neg_activations.float(), []


def compute_steering_vectors(
    persona_id: str,
    model_name: str,
    layers: list[int] | None = None,
    activations_dir: Path | str | None = None,
    method: Literal["mean", "pca"] = "mean",
    center: bool = True,
    verbose: bool = True,
) -> dict:
    """Compute per-layer steering vectors for a persona.

    Args:
        persona_id: Persona UUID.
        model_name: HF model name (used to locate activation directory).
        layers: Layer indices to compute vectors for. None = all layers.
        activations_dir: Root directory for activations. Defaults to
            artifacts/activations.
        method: "mean" (CAA-style) or "pca" (first right singular vector).
        center: If True, subtract each activation's feature-dimension mean
            before computing diffs.
        verbose: If True, print a compact summary table.

    Returns:
        Dict with:
            steering_vectors  (n_layers, hidden_dim) — one vector per layer
            suggested_alphas  (n_layers,) — per-layer calibrated alpha
            persona_gaps      (n_layers,) — L2 norm of mean diff per layer
            layer_indices     list[int]
            persona_id, model_id, n_qa_pairs, method, center
    """
    store = ActivationStore(model_name, activations_dir)
    loaded = _load_pos_neg(store, persona_id, verbose)
    if loaded is None:
        return {}

    pos_activations, neg_activations, questions = loaded
    layer_indices = layers if layers is not None else list(range(pos_activations.shape[0]))

    all_svs: list[torch.Tensor] = []
    all_alphas: list[float] = []
    all_gaps: list[float] = []

    for layer_idx in layer_indices:
        raw_sv, sv_norm, suggested_alpha = _compute_layer_sv(
            pos_activations[layer_idx, :].unsqueeze(0),
            neg_activations[layer_idx, :].unsqueeze(0),
            method,
            center,
        )
        all_svs.append(raw_sv)
        all_alphas.append(suggested_alpha)
        all_gaps.append(sv_norm)

    steering_vectors = torch.stack(all_svs, dim=0)  # (n_layers, hidden_dim)
    suggested_alphas = torch.tensor(all_alphas)
    persona_gaps = torch.tensor(all_gaps)

    if verbose:
        peak_i = int(persona_gaps.argmax().item())
        table = Table(title=f"Steering Vectors — {persona_id[:8]}…")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_row("Layers computed", str(len(layer_indices)))
        table.add_row("Method / Center", f"{method} / {center}")
        table.add_row("Peak gap layer", str(layer_indices[peak_i]))
        table.add_row("Peak persona gap", f"{persona_gaps[peak_i].item():.4f}")
        table.add_row("Mean suggested alpha", f"{suggested_alphas.mean().item():.4f}")
        table.add_row("QA pairs", str(len(questions)))
        console.print(table)

    return {
        "steering_vectors": steering_vectors,
        "suggested_alphas": suggested_alphas,
        "persona_gaps": persona_gaps,
        "layer_indices": layer_indices,
        "persona_id": persona_id,
        "model_id": model_name,
        "n_qa_pairs": len(questions),
        "method": method,
        "center": center,
    }


def compute_cross_persona_steering_vectors(
    all_persona_ids: list[str],
    model_name: str,
    activations_dir: Path | str | None = None,
    mask_strategy: object | None = None,
    method: Literal["mean", "pca"] = "mean",
    center: bool = True,
    verbose: bool = True,
    store=None,
) -> dict[str, dict]:
    """Compute steering vectors using leave-one-out cross-persona mean as negative.

    For persona A:
        positive = A's biography activations  (num_layers, hidden_dim)
        negative = mean biography activations of all other personas

    The leave-one-out mean removes the shared "biography mode" direction
    (format, length, position) and isolates each persona's distinct signal.

    Args:
        all_persona_ids: All persona IDs to include. Each is the target in turn.
        model_name: HF model name.
        activations_dir: Root activations directory. Ignored when store is provided.
        method: "mean" or "pca".
        center: Feature-dimension centering before diffs.
        verbose: Print a summary table when done.
        store: Optional pre-built store (ActivationStore or HFActivationStore).
            Overrides activations_dir when provided.

    Returns:
        {persona_id: sv_dict} — same format as compute_steering_vectors output.
    """
    if store is None:
        store = ActivationStore(model_name, activations_dir, mask_strategy=mask_strategy)

    # Load biography activations for every persona that has them
    persona_mean_bio: dict[str, torch.Tensor] = {}   # (n_layers, hidden_dim) per persona
    persona_full_bio: dict[str, torch.Tensor] = {}

    for pid in all_persona_ids:
        try:
            acts = store.load("biography", pid).float()    # (n_layers, hidden_dim)
            persona_mean_bio[pid] = acts
            persona_full_bio[pid] = acts
        except FileNotFoundError:
            pass

    loaded_ids = list(persona_mean_bio.keys())
    n = len(loaded_ids)
    if n < 2:
        if verbose:
            console.print("[yellow]Need ≥2 personas with biography activations.[/]")
        return {}

    # Global sum enables O(1) leave-one-out: mean_others = (sum_all - this) / (n-1)
    stacked = torch.stack([persona_mean_bio[p] for p in loaded_ids], dim=0)  # (n, n_layers, hidden_dim)
    global_sum = stacked.sum(dim=0)                                           # (n_layers, hidden_dim)

    results: dict[str, dict] = {}
    for pid in loaded_ids:
        pos_acts = persona_full_bio[pid]                   # (n_layers, hidden_dim)
        n_layers = pos_acts.shape[0]

        # Leave-one-out negative: mean of all other personas' biography activations
        other_mean = (global_sum - persona_mean_bio[pid]) / (n - 1)   # (n_layers, hidden_dim)

        all_svs: list[torch.Tensor] = []
        all_alphas: list[float] = []
        all_gaps: list[float] = []

        for layer_idx in range(n_layers):
            pos_stack = pos_acts[layer_idx, :].unsqueeze(0)    # (1, hidden_dim)
            neg_stack = other_mean[layer_idx, :].unsqueeze(0)  # (1, hidden_dim)

            raw_sv, sv_norm, suggested_alpha = _compute_layer_sv(
                pos_stack, neg_stack, method, center
            )
            all_svs.append(raw_sv)
            all_alphas.append(suggested_alpha)
            all_gaps.append(sv_norm)

        results[pid] = {
            "steering_vectors": torch.stack(all_svs, dim=0),
            "suggested_alphas": torch.tensor(all_alphas),
            "persona_gaps": torch.tensor(all_gaps),
            "layer_indices": list(range(n_layers)),
            "persona_id": pid,
            "model_id": model_name,
            "n_qa_pairs": 1,
            "method": method,
            "center": center,
        }

    if verbose:
        table = Table(title=f"Cross-Persona Steering Vectors ({n} personas)")
        table.add_column("Persona ID", style="cyan")
        table.add_column("Peak gap layer", justify="right", style="magenta")
        table.add_column("Peak gap", justify="right", style="magenta")
        table.add_column("Mean α", justify="right", style="dim")
        for pid in loaded_ids:
            gaps = results[pid]["persona_gaps"]
            alphas = results[pid]["suggested_alphas"]
            peak_i = int(gaps.argmax().item())
            table.add_row(
                pid[:8] + "…",
                str(peak_i),
                f"{gaps[peak_i].item():.4f}",
                f"{alphas.mean().item():.4f}",
            )
        console.print(table)

    return results


def compute_steering_vector(
    persona_id: str,
    model_name: str,
    layer_idx: int = STEER_LAYER,
    mask_strategy: object | None = None,
    activations_dir: Path | str | None = None,
    method: Literal["mean", "pca"] = "mean",
    center: bool = True,
    verbose: bool = True,
) -> dict:
    """Single-layer wrapper around compute_steering_vectors.

    Kept for backward compatibility with notebooks and the steer CLI command.
    For multi-layer steering use compute_steering_vectors directly.

    Returns:
        Dict with steering_vector (1, 1, hidden_dim), suggested_alpha, and metadata.
    """
    store = ActivationStore(model_name, activations_dir)

    if verbose:
        print(f"\nLoading activations for {persona_id}...")

    # Load both positive (biography) and negative (templated) activation means.
    try:
        pos_activations = store.load(
            "biography", persona_id, mask_strategy=mask_strategy
        )
    except FileNotFoundError:
        if verbose:
            print("✗ Biography activations not found. Run extraction first.")
        return {}

    try:
        neg_activations = store.load(
            "templated", persona_id, mask_strategy=mask_strategy
        )
    except FileNotFoundError:
        if verbose:
            print("✗ Templated activations not found. Run extraction first.")
        return {}

    # Verify shape alignment.
    if pos_activations.shape != neg_activations.shape:
        raise ValueError(
            "Mismatch: "
            f"{tuple(pos_activations.shape)} positive vs "
            f"{tuple(neg_activations.shape)} negative"
        )
    if layer_idx < 0 or layer_idx >= pos_activations.shape[0]:
        raise ValueError(
            f"layer_idx {layer_idx} is out of range for {pos_activations.shape[0]} layers"
        )

    if verbose:
        print("Computing steering direction...")
    pos_mean = pos_activations[layer_idx, :].float()
    neg_mean = neg_activations[layer_idx, :].float()
    raw_sv = pos_mean - neg_mean

    steering_vector = raw_sv.unsqueeze(0).unsqueeze(0)

    mean_rms = neg_mean.pow(2).mean().sqrt().item()
    sv_norm = raw_sv.norm().item()
    suggested_alpha = (20.0 * mean_rms) / (sv_norm + 1e-8)
    steering_vector = raw_sv.unsqueeze(0).unsqueeze(0)  # (1, 1, hidden_dim)

    if verbose:
        table = Table(title="Steering Vector Summary")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_row("Persona ID", persona_id)
        table.add_row("Layer", str(layer_idx))
        table.add_row("Method", method)
        table.add_row("Center", str(center))
        table.add_row("Shape", str(tuple(steering_vector.shape)))
        table.add_row("L2 Norm", f"{sv_norm:.6f}")
        table.add_row("Suggested Alpha", f"{suggested_alpha:.4f}")
        table.add_row("Hidden Size", str(pos_activations.shape[1]))
        console.print(table)

    return {
        "steering_vector": steering_vector,
        "suggested_alpha": suggested_alpha,
        "persona_id": persona_id,
        "layer": layer_idx,
        "model_id": model_name,
        "hidden_size": int(pos_activations.shape[1]),
    }


def save_steering_vectors(
    sv_dict: dict, out_path: str | Path, verbose: bool = True
) -> Path:
    """Save multi-layer steering vectors to safetensors + metadata.json."""
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    save_file(
        {
            "steering_vectors": sv_dict["steering_vectors"].detach().cpu(),
            "suggested_alphas": sv_dict["suggested_alphas"].detach().cpu(),
            "persona_gaps": sv_dict["persona_gaps"].detach().cpu(),
        },
        str(out_path / "steering_vectors.safetensors"),
    )

    metadata = {
        "layer_indices": sv_dict["layer_indices"],
        "persona_id": sv_dict["persona_id"],
        "model_id": sv_dict["model_id"],
        "n_qa_pairs": sv_dict["n_qa_pairs"],
        "method": sv_dict.get("method", "mean"),
        "center": sv_dict.get("center", True),
    }
    (out_path / "metadata.json").write_text(json.dumps(metadata, indent=2))

    if verbose:
        print(f"\nSaved {len(sv_dict['layer_indices'])} layer vectors to {out_path}")
    return out_path


def load_steering_vectors(path: str | Path) -> dict:
    """Load multi-layer steering vectors from safetensors + metadata.json."""
    path = Path(path)
    tensors = load_file(str(path / "steering_vectors.safetensors"))
    metadata = json.loads((path / "metadata.json").read_text())
    return {
        "steering_vectors": tensors["steering_vectors"],
        "suggested_alphas": tensors["suggested_alphas"],
        "persona_gaps": tensors["persona_gaps"],
        **metadata,
    }


def save_steering_vector(
    sv_dict: dict, out_path: str | Path, verbose: bool = True
) -> Path:
    """Save single-layer steering vector dict to safetensors + metadata.json."""
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    save_file(
        {"steering_vector": sv_dict["steering_vector"].detach().cpu()},
        str(out_path / "steering_vector.safetensors"),
    )

    metadata = {
        "suggested_alpha": sv_dict["suggested_alpha"],
        "persona_id": sv_dict["persona_id"],
        "layer": sv_dict["layer"],
        "model_id": sv_dict["model_id"],
        "hidden_size": sv_dict["hidden_size"],
    }
    (out_path / "metadata.json").write_text(json.dumps(metadata, indent=2))

    if verbose:
        print(f"\nSaved to {out_path}")
    return out_path


def load_steering_vector(path: str | Path) -> dict:
    """Load a saved single-layer steering vector from safetensors + metadata.json."""
    path = Path(path)
    tensors = load_file(str(path / "steering_vector.safetensors"))
    metadata = json.loads((path / "metadata.json").read_text())
    return {
        "steering_vector": tensors["steering_vector"],
        **metadata,
    }
