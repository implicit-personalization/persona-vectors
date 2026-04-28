"""
Generate persona steering vectors from pre-extracted activations.

Method: Contrastive mean-diff
──────────────────────────────────────────────────────────────────────────────
For each QA pair:

  negative prompt  →  Templated prompt + Question + Answer
  positive prompt  →  Biography + Question + Answer

Extract the MEAN of the RESPONSE TOKENS' hidden states at STEER_LAYER across
all QA pairs for the persona.

  steering_vector = mean_over_questions(biography_h) - mean_over_questions(templated_h)

Output saved to: artifacts/vectors/{persona_id}.safetensors
"""

import json
from pathlib import Path

import torch
from rich.console import Console
from rich.table import Table
from safetensors.torch import load_file, save_file
from tqdm import tqdm

from persona_vectors.artifacts import ActivationStore

# Config
STEER_LAYER = 20

console = Console()


def compute_steering_vector(
    persona_id: str,
    model_name: str,
    layer_idx: int = STEER_LAYER,
    mask_strategy: object | None = None,
    activations_dir: Path | str | None = None,
    verbose: bool = True,
) -> dict:
    """Load saved activations and compute steering vector.

    Uses pre-extracted masked-mean activations (response tokens only) from
    both biography and templated prompt variants.

    Args:
        persona_id: Persona UUID.
        model_name: HF model name (used to locate activation directory).
        layer_idx: Layer index for the steering vector.
        activations_dir: Root directory for activations. Defaults to
            artifacts/activations.
        verbose: If True, print progress and summary table.

    Returns:
        Dict with steering_vector, suggested_alpha, and metadata.
    """
    store = ActivationStore(model_name, activations_dir)

    if verbose:
        print(f"\nLoading activations for {persona_id}...")

    # Load both positive (biography) and negative (templated) activations
    try:
        pos_activations, pos_sample_ids = store.load(
            "biography", persona_id, mask_strategy=mask_strategy
        )
    except FileNotFoundError:
        if verbose:
            print("✗ Biography activations not found. Run extraction first.")
        return {}

    try:
        neg_activations, neg_sample_ids = store.load(
            "templated", persona_id, mask_strategy=mask_strategy
        )
    except FileNotFoundError:
        if verbose:
            print("✗ Templated activations not found. Run extraction first.")
        return {}

    # Verify alignment
    if len(pos_activations) != len(neg_activations):
        raise ValueError(
            f"Mismatch: {len(pos_activations)} positive vs {len(neg_activations)} negative"
        )

    pos_vectors = []
    neg_vectors = []

    if verbose:
        print("Computing response-token means...")
    for i, (pos_act, neg_act) in enumerate(
        tqdm(zip(pos_activations, neg_activations), disable=not verbose)
    ):
        if pos_sample_ids[i] != neg_sample_ids[i]:
            print(f"Warning: sample id mismatch at index {i}")

        # pos_act shape: [num_layers, hidden_size] (already masked mean over response tokens)
        # Extract the layer we care about
        pos_mean = pos_act[layer_idx, :]  # [hidden_size]
        neg_mean = neg_act[layer_idx, :]  # [hidden_size]

        pos_vectors.append(pos_mean)
        neg_vectors.append(neg_mean)

    # Compute steering vector
    pos_stack = torch.stack(pos_vectors)  # [n_questions, hidden_size]
    neg_stack = torch.stack(neg_vectors)

    mean_pos = pos_stack.mean(dim=0)
    mean_neg = neg_stack.mean(dim=0)

    raw_sv = mean_pos - mean_neg

    # Steering vector shape: [1, 1, hidden_dim]
    steering_vector = raw_sv.unsqueeze(0).unsqueeze(0)

    # Calculate Alpha (20x Mean RMS of negatives)
    mean_rms = neg_stack.pow(2).mean(dim=-1).sqrt().mean().item()
    sv_norm = raw_sv.norm().item()
    suggested_alpha = (20.0 * mean_rms) / (sv_norm + 1e-8)

    if verbose:
        table = Table(title="Steering Vector Summary")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_row("Persona ID", persona_id)
        table.add_row("Layer", str(layer_idx))
        table.add_row("Shape", str(tuple(steering_vector.shape)))
        table.add_row("L2 Norm", f"{sv_norm:.6f}")
        table.add_row("Mean Neg RMS", f"{mean_rms:.6f}")
        table.add_row("Suggested Alpha", f"{suggested_alpha:.4f}")
        table.add_row("QA Pairs Used", str(len(pos_vectors)))
        console.print(table)

    return {
        "steering_vector": steering_vector,
        "suggested_alpha": suggested_alpha,
        "persona_id": persona_id,
        "layer": layer_idx,
        "model_id": model_name,
        "n_qa_pairs": len(pos_vectors),
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

    # Save metadata as JSON
    metadata = {
        "suggested_alpha": sv_dict["suggested_alpha"],
        "persona_id": sv_dict["persona_id"],
        "layer": sv_dict["layer"],
        "model_id": sv_dict["model_id"],
        "n_qa_pairs": sv_dict["n_qa_pairs"],
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
