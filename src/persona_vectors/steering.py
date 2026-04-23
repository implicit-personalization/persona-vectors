"""
Generate persona steering vectors from pre-extracted activations.

Method: Contrastive mean-diff (default) or PCA of per-question diffs
──────────────────────────────────────────────────────────────────────────────
For each QA pair:

  negative prompt  →  {templated | baseline} prompt + Question + Answer
  positive prompt  →  Biography + Question + Answer

Compute per-question diffs:  d_i = biography_h_i - negative_h_i

  method="mean":  steering_vector = mean(d_i)
  method="pca":   for each selected layer, take the first right singular
                  vector of the centered diff matrix, sign-aligned with
                  mean(d_i) and scaled to match its L2 norm.

Centering (center=True):
  Before computing diffs, each per-question hidden vector is centered by
  subtracting its own feature-dimension mean, removing the per-vector DC
  offset shared across prompt variants.

Output saved to: artifacts/vectors/{persona_id}/
"""

import json
from pathlib import Path
from typing import Literal

import torch
from rich.console import Console
from rich.table import Table
from safetensors.torch import load_file, save_file

from persona_vectors.artifacts import ActivationStore, list_personas

# Config
STEER_LAYER = 20

console = Console()

NegativeVariant = Literal["templated", "baseline", "pooled_biography"]
SteeringMethod = Literal["mean", "pca"]


def _shared_item_key(
    *,
    qid: str | None,
    question: str,
    persona_id: str,
) -> str:
    if qid and qid.startswith(f"{persona_id}-"):
        return qid[len(persona_id) + 1 :]
    return qid or question


def _load_pooled_biography_negative(
    *,
    store: ActivationStore,
    persona_id: str,
    pos_activations: torch.Tensor,
    pos_qids: list[str] | None,
    pos_questions: list[str],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | int]]:
    available_personas = [
        pid
        for pid in list_personas(store.root_dir, store.model_name, ["biography"])
        if pid != persona_id
    ]
    if not available_personas:
        raise ValueError("No other biography activations available for pooled contrast")

    negative_bank: dict[str, list[torch.Tensor]] = {}
    for other_id in available_personas:
        try:
            other_activations, other_qids, other_questions = store.load_records(
                "biography", other_id
            )
        except FileNotFoundError:
            continue

        for idx in range(other_activations.shape[0]):
            key = _shared_item_key(
                qid=other_qids[idx] if other_qids is not None else None,
                question=other_questions[idx],
                persona_id=other_id,
            )
            negative_bank.setdefault(key, []).append(other_activations[idx].float())

    matched_pos: list[torch.Tensor] = []
    matched_neg: list[torch.Tensor] = []
    negative_counts: list[int] = []
    for idx in range(pos_activations.shape[0]):
        key = _shared_item_key(
            qid=pos_qids[idx] if pos_qids is not None else None,
            question=pos_questions[idx],
            persona_id=persona_id,
        )
        pooled = negative_bank.get(key)
        if not pooled:
            continue
        matched_pos.append(pos_activations[idx].float())
        matched_neg.append(torch.stack(pooled, dim=0).mean(dim=0))
        negative_counts.append(len(pooled))

    if not matched_pos:
        raise ValueError("No shared-item overlap found for pooled biography contrast")

    pos_stack = torch.stack(matched_pos, dim=0)
    neg_stack = torch.stack(matched_neg, dim=0)
    stats = {
        "matched_items": len(matched_pos),
        "available_other_personas": len(available_personas),
        "mean_other_personas_per_item": float(sum(negative_counts) / len(negative_counts)),
        "min_other_personas_per_item": int(min(negative_counts)),
        "max_other_personas_per_item": int(max(negative_counts)),
    }
    return pos_stack, neg_stack, stats


def _normalize_layers(num_layers: int, layer_idx: int | list[int] | None) -> list[int]:
    if layer_idx is None:
        return list(range(num_layers))
    if isinstance(layer_idx, int):
        layers = [layer_idx]
    else:
        layers = list(layer_idx)
    if not layers:
        raise ValueError("layer_idx must select at least one layer")
    if min(layers) < 0 or max(layers) >= num_layers:
        raise ValueError(
            f"Requested layers {layers} outside valid range [0, {num_layers - 1}]"
        )
    return layers


def _compute_pca_direction(diffs: torch.Tensor) -> torch.Tensor:
    """Compute one steering direction per selected layer from per-question diffs."""
    directions: list[torch.Tensor] = []
    for layer_diffs in diffs.unbind(dim=1):
        centered = layer_diffs - layer_diffs.mean(dim=0, keepdim=True)
        _, _, vt = torch.linalg.svd(centered, full_matrices=False)
        direction = vt[0]
        mean_diff = layer_diffs.mean(dim=0)
        if torch.dot(direction, mean_diff) < 0:
            direction = -direction
        directions.append(direction * mean_diff.norm())
    return torch.stack(directions, dim=0)


def compute_steering_vector(
    persona_id: str,
    model_name: str,
    layer_idx: int | list[int] | None = STEER_LAYER,
    activations_dir: Path | str | None = None,
    negative_variant: NegativeVariant = "templated",
    method: SteeringMethod = "mean",
    center: bool = False,
    verbose: bool = True,
) -> dict:
    """Load saved activations and compute steering vector.

    Uses pre-extracted masked-mean activations (response tokens only) from
    the biography prompt and a configurable negative prompt variant.

    Args:
        persona_id: Persona UUID.
        model_name: HF model name (used to locate activation directory).
        layer_idx: Layer index (or indices) for the steering vector. ``None``
            means all layers present in the saved activations.
        activations_dir: Root directory for activations. Defaults to
            artifacts/activations.
        negative_variant: Which contrast prompt to subtract from biography.
            ``"baseline"`` is the stricter CAA-style control; ``"templated"``
            preserves the older behavior.
        method: How to aggregate per-question diffs into a steering direction.
            ``"mean"`` reproduces the original mean-diff path.
            ``"pca"`` keeps the dominant per-layer diff direction.
        center: If True, subtract each activation's own feature-dimension mean
            before computing diffs.
        verbose: If True, print progress and summary table.

    Returns:
        Dict with steering_vector, suggested_alpha, and metadata.
    """
    store = ActivationStore(model_name, activations_dir)

    if verbose:
        print(f"\nLoading activations for {persona_id}...")

    # Load both positive (biography) and negative activations.
    try:
        pos_activations, pos_qids, pos_questions = store.load_records(
            "biography", persona_id
        )
    except FileNotFoundError:
        if verbose:
            print("✗ Biography activations not found. Run extraction first.")
        return {}

    pooled_stats: dict[str, float | int] = {}
    if negative_variant == "pooled_biography":
        pos_stack, neg_stack, pooled_stats = _load_pooled_biography_negative(
            store=store,
            persona_id=persona_id,
            pos_activations=pos_activations,
            pos_qids=pos_qids,
            pos_questions=pos_questions,
        )
    else:
        try:
            neg_activations, neg_qids, neg_questions = store.load_records(
                negative_variant, persona_id
            )
        except FileNotFoundError:
            if verbose:
                print(f"✗ {negative_variant.capitalize()} activations not found. Run extraction first.")
            return {}

        if len(pos_activations) != len(neg_activations):
            raise ValueError(
                f"Mismatch: {len(pos_activations)} positive vs {len(neg_activations)} negative"
            )

        if pos_activations.ndim != 3 or neg_activations.ndim != 3:
            raise ValueError("Expected activation tensors with shape (n_questions, num_layers, hidden_size)")

        for i in range(len(pos_activations)):
            if pos_qids is not None and neg_qids is not None and pos_qids[i] != neg_qids[i]:
                raise ValueError(
                    f"QID mismatch at index {i}: {pos_qids[i]!r} vs {neg_qids[i]!r}"
                )
            if pos_questions[i] != neg_questions[i]:
                raise ValueError(
                    f"Question mismatch at index {i}: {pos_questions[i]!r} vs {neg_questions[i]!r}"
                )

        pos_stack = pos_activations.float()
        neg_stack = neg_activations.float()

    selected_layers = _normalize_layers(pos_stack.shape[1], layer_idx)
    pos_stack = pos_stack[:, selected_layers, :].float()
    neg_stack = neg_stack[:, selected_layers, :].float()

    if center:
        pos_stack = pos_stack - pos_stack.mean(dim=-1, keepdim=True)
        neg_stack = neg_stack - neg_stack.mean(dim=-1, keepdim=True)

    diffs = pos_stack - neg_stack
    if method == "mean":
        raw_sv = diffs.mean(dim=0)
    elif method == "pca":
        raw_sv = _compute_pca_direction(diffs)
    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'mean' or 'pca'.")

    steering_vector = raw_sv.unsqueeze(0)

    mean_rms_by_layer = neg_stack.pow(2).mean(dim=-1).sqrt().mean(dim=0)
    sv_norm_by_layer = raw_sv.norm(dim=-1)
    suggested_alpha_by_layer = (20.0 * mean_rms_by_layer) / (sv_norm_by_layer + 1e-8)
    suggested_alpha = float(suggested_alpha_by_layer.mean().item())

    if verbose:
        table = Table(title="Steering Vector Summary")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_row("Persona ID", persona_id)
        table.add_row(
            "Layers",
            str(selected_layers[0]) if len(selected_layers) == 1 else f"{selected_layers[0]}-{selected_layers[-1]} ({len(selected_layers)})",
        )
        table.add_row("Negative Variant", negative_variant)
        table.add_row("Method", method)
        table.add_row("Center", str(center))
        table.add_row("Shape", str(tuple(steering_vector.shape)))
        if pooled_stats:
            table.add_row("Matched Shared Items", str(pooled_stats["matched_items"]))
            table.add_row(
                "Other Personas / Item",
                f"{pooled_stats['mean_other_personas_per_item']:.2f} avg"
                f" ({pooled_stats['min_other_personas_per_item']}-{pooled_stats['max_other_personas_per_item']})",
            )
        if len(selected_layers) == 1:
            table.add_row("L2 Norm", f"{float(sv_norm_by_layer[0].item()):.6f}")
            table.add_row("Mean Neg RMS", f"{float(mean_rms_by_layer[0].item()):.6f}")
        else:
            table.add_row(
                "L2 Norm Range",
                f"{float(sv_norm_by_layer.min().item()):.6f} - {float(sv_norm_by_layer.max().item()):.6f}",
            )
            table.add_row(
                "Mean Neg RMS Range",
                f"{float(mean_rms_by_layer.min().item()):.6f} - {float(mean_rms_by_layer.max().item()):.6f}",
            )
        table.add_row("Suggested Alpha", f"{suggested_alpha:.4f}")
        table.add_row("QA Pairs Used", str(pos_stack.shape[0]))
        console.print(table)

    return {
        "steering_vector": steering_vector,
        "suggested_alpha": suggested_alpha,
        "suggested_alpha_by_layer": [float(v) for v in suggested_alpha_by_layer.tolist()],
        "persona_id": persona_id,
        "layer": selected_layers[0] if len(selected_layers) == 1 else None,
        "layers": selected_layers,
        "model_id": model_name,
        "n_qa_pairs": int(pos_stack.shape[0]),
        "negative_variant": negative_variant,
        "method": method,
        "center": center,
        **pooled_stats,
    }


def save_steering_vector(
    sv_dict: dict, out_path: str | Path, verbose: bool = True
) -> Path:
    """Save steering vector dict to safetensors + metadata.json.

    Args:
        sv_dict: Dict from compute_steering_vector.
        out_path: Directory to save into. Creates {persona_id}.safetensors
            and metadata.json inside.
        verbose: If True, print confirmation.

    Returns:
        Path to the output directory.
    """
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    # Save tensor as safetensors
    tensor_path = out_path / "steering_vector.safetensors"
    save_file(
        {"steering_vector": sv_dict["steering_vector"].detach().cpu()},
        str(tensor_path),
    )

    layer_value = sv_dict.get("layer")
    metadata = {
        "suggested_alpha": sv_dict["suggested_alpha"],
        "suggested_alpha_by_layer": sv_dict.get("suggested_alpha_by_layer"),
        "persona_id": sv_dict["persona_id"],
        "layer": layer_value,
        "layers": sv_dict.get("layers"),
        "model_id": sv_dict["model_id"],
        "n_qa_pairs": sv_dict["n_qa_pairs"],
        "negative_variant": sv_dict.get("negative_variant", "templated"),
        "method": sv_dict.get("method", "mean"),
        "center": sv_dict.get("center", False),
    }
    metadata_path = out_path / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    if verbose:
        print(f"\nSaved to {out_path}")
    return out_path


def load_steering_vector(path: str | Path) -> dict:
    """Load a saved steering vector from safetensors + metadata.json.

    Args:
        path: Directory containing steering_vector.safetensors and metadata.json.

    Returns:
        Dict with steering_vector tensor and metadata fields.
    """
    path = Path(path)

    tensors = load_file(str(path / "steering_vector.safetensors"))
    metadata = json.loads((path / "metadata.json").read_text())

    return {
        "steering_vector": tensors["steering_vector"],
        **metadata,
    }
