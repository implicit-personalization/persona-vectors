import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def _model_dir_name(model_name: str) -> str:
    return model_name.replace("/", "__")


def build_activation_path(
    root_dir: str | Path,
    model_name: str,
    prompt_variant: str,
    persona_id: str,
) -> Path:
    return Path(root_dir) / _model_dir_name(model_name) / prompt_variant / persona_id


def _save_metadata(path: Path, updates: dict) -> None:
    existing = {}
    if path.exists():
        existing = json.loads(path.read_text())
    existing.update(updates)
    path.write_text(json.dumps(existing, indent=2))


def save_per_question_vectors(
    root_dir: str | Path,
    model_name: str,
    prompt_variant: str,
    persona_id: str,
    per_question_vectors: torch.Tensor,
    questions: list[str],
) -> Path:
    if per_question_vectors.ndim != 3:
        raise ValueError(
            "per_question_vectors must have shape (n_questions, n_layers, d_model)"
        )

    if len(questions) != per_question_vectors.shape[0]:
        raise ValueError("number of questions must match first tensor dimension")

    artifact_dir = build_activation_path(
        root_dir=root_dir,
        model_name=model_name,
        prompt_variant=prompt_variant,
        persona_id=persona_id,
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)

    tensor_path = artifact_dir / "activations.safetensors"
    tensors = {"per_question_vectors": per_question_vectors.detach().cpu()}
    save_file(tensors, str(tensor_path))

    metadata_path = artifact_dir / "metadata.json"
    metadata = {"questions": questions}
    metadata_path.write_text(json.dumps(metadata, indent=2))

    return artifact_dir


def load_per_question_vectors(
    root_dir: str | Path,
    model_name: str,
    prompt_variant: str,
    persona_id: str,
) -> tuple[torch.Tensor, list[str]]:
    artifact_dir = build_activation_path(
        root_dir=root_dir,
        model_name=model_name,
        prompt_variant=prompt_variant,
        persona_id=persona_id,
    )

    tensors = load_file(str(artifact_dir / "activations.safetensors"))
    metadata = json.loads((artifact_dir / "metadata.json").read_text())

    return tensors["per_question_vectors"], metadata["questions"]


def save_per_prompt_summaries(
    root_dir: str | Path,
    model_name: str,
    prompt_variant: str,
    persona_id: str,
    neutral_prompts: list[str],
    response_mean_per_prompt: torch.Tensor,
    last_prompt_per_prompt: torch.Tensor,
    metadata: dict | None = None,
) -> Path:
    """Save per-neutral-prompt summaries with shape (n_prompts, n_layers, d_model)."""
    if response_mean_per_prompt.ndim != 3 or last_prompt_per_prompt.ndim != 3:
        raise ValueError("per-prompt summaries must each have shape (n_prompts, n_layers, d_model)")
    if response_mean_per_prompt.shape != last_prompt_per_prompt.shape:
        raise ValueError("response_mean_per_prompt and last_prompt_per_prompt must match shape")
    if response_mean_per_prompt.shape[0] != len(neutral_prompts):
        raise ValueError("number of neutral prompts must match first tensor dimension")

    artifact_dir = build_activation_path(
        root_dir=root_dir,
        model_name=model_name,
        prompt_variant=prompt_variant,
        persona_id=persona_id,
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)

    save_file(
        {"response_mean_per_prompt": response_mean_per_prompt.detach().cpu()},
        str(artifact_dir / "response_mean_per_prompt.safetensors"),
    )
    save_file(
        {"last_prompt_per_prompt": last_prompt_per_prompt.detach().cpu()},
        str(artifact_dir / "last_prompt_per_prompt.safetensors"),
    )

    metadata_path = artifact_dir / "metadata.json"
    _save_metadata(
        metadata_path,
        {
            "neutral_prompts": neutral_prompts,
            "summary_shape": list(response_mean_per_prompt.shape),
            **(metadata or {}),
        },
    )
    return artifact_dir


def load_per_prompt_summaries(
    root_dir: str | Path,
    model_name: str,
    prompt_variant: str,
    persona_id: str,
) -> tuple[torch.Tensor, torch.Tensor, list[str], dict]:
    """Load per-neutral-prompt summaries and metadata."""
    artifact_dir = build_activation_path(
        root_dir=root_dir,
        model_name=model_name,
        prompt_variant=prompt_variant,
        persona_id=persona_id,
    )
    response = load_file(str(artifact_dir / "response_mean_per_prompt.safetensors"))[
        "response_mean_per_prompt"
    ]
    last = load_file(str(artifact_dir / "last_prompt_per_prompt.safetensors"))[
        "last_prompt_per_prompt"
    ]
    metadata = json.loads((artifact_dir / "metadata.json").read_text())
    return response, last, metadata["neutral_prompts"], metadata


def save_persona_representations(
    root_dir: str | Path,
    model_name: str,
    prompt_variant: str,
    persona_id: str,
    persona_response_mean: torch.Tensor,
    persona_last_prompt: torch.Tensor,
    metadata: dict | None = None,
) -> Path:
    """Save averaged persona representations with shape (n_layers, d_model)."""
    if persona_response_mean.ndim != 2 or persona_last_prompt.ndim != 2:
        raise ValueError("persona representations must each have shape (n_layers, d_model)")
    if persona_response_mean.shape != persona_last_prompt.shape:
        raise ValueError("persona_response_mean and persona_last_prompt must match shape")

    artifact_dir = build_activation_path(
        root_dir=root_dir,
        model_name=model_name,
        prompt_variant=prompt_variant,
        persona_id=persona_id,
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)

    save_file(
        {"persona_response_mean": persona_response_mean.detach().cpu()},
        str(artifact_dir / "persona_response_mean.safetensors"),
    )
    save_file(
        {"persona_last_prompt": persona_last_prompt.detach().cpu()},
        str(artifact_dir / "persona_last_prompt.safetensors"),
    )

    _save_metadata(
        artifact_dir / "metadata.json",
        {
            "persona_representation_shape": list(persona_response_mean.shape),
            **(metadata or {}),
        },
    )
    return artifact_dir


def load_persona_representations(
    root_dir: str | Path,
    model_name: str,
    prompt_variant: str,
    persona_id: str,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Load averaged persona representations and metadata."""
    artifact_dir = build_activation_path(
        root_dir=root_dir,
        model_name=model_name,
        prompt_variant=prompt_variant,
        persona_id=persona_id,
    )
    response = load_file(str(artifact_dir / "persona_response_mean.safetensors"))[
        "persona_response_mean"
    ]
    last = load_file(str(artifact_dir / "persona_last_prompt.safetensors"))[
        "persona_last_prompt"
    ]
    metadata = json.loads((artifact_dir / "metadata.json").read_text())
    return response, last, metadata


def save_contrastive_vectors(
    root_dir: str | Path,
    model_name: str,
    prompt_variant: str,
    persona_id: str,
    contrastive_response_mean: torch.Tensor,
    contrastive_last_prompt: torch.Tensor,
    metadata: dict | None = None,
) -> Path:
    """Save final contrastive vectors with shape (n_layers, d_model)."""
    if contrastive_response_mean.ndim != 2 or contrastive_last_prompt.ndim != 2:
        raise ValueError("contrastive vectors must each have shape (n_layers, d_model)")
    if contrastive_response_mean.shape != contrastive_last_prompt.shape:
        raise ValueError(
            "contrastive_response_mean and contrastive_last_prompt must match shape"
        )

    artifact_dir = build_activation_path(
        root_dir=root_dir,
        model_name=model_name,
        prompt_variant=prompt_variant,
        persona_id=persona_id,
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)

    save_file(
        {"contrastive_response_mean": contrastive_response_mean.detach().cpu()},
        str(artifact_dir / "contrastive_response_mean.safetensors"),
    )
    save_file(
        {"contrastive_last_prompt": contrastive_last_prompt.detach().cpu()},
        str(artifact_dir / "contrastive_last_prompt.safetensors"),
    )

    _save_metadata(
        artifact_dir / "metadata.json",
        {
            "contrastive_shape": list(contrastive_response_mean.shape),
            **(metadata or {}),
        },
    )
    return artifact_dir


def load_contrastive_vectors(
    root_dir: str | Path,
    model_name: str,
    prompt_variant: str,
    persona_id: str,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Load final contrastive vectors and metadata."""
    artifact_dir = build_activation_path(
        root_dir=root_dir,
        model_name=model_name,
        prompt_variant=prompt_variant,
        persona_id=persona_id,
    )
    response = load_file(str(artifact_dir / "contrastive_response_mean.safetensors"))[
        "contrastive_response_mean"
    ]
    last = load_file(str(artifact_dir / "contrastive_last_prompt.safetensors"))[
        "contrastive_last_prompt"
    ]
    metadata = json.loads((artifact_dir / "metadata.json").read_text())
    return response, last, metadata
