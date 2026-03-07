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
