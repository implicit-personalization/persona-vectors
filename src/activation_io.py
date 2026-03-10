"""Per-question activation storage and retrieval.
The storage format uses INDEX-BASED MAPPING:
    tensor at index i  <->  metadata[i]  <->  metadata[i]["qid"]

Example for a persona with 3 questions:
    per_question_activations[0]  <->  {"qid": "explicit_0", ...}
    per_question_activations[1]  <->  {"qid": "implicit_0", ...}
    per_question_activations[2]  <->  {"qid": "explicit_1", ...}

This means:
1. When saving: tensor list order must match metadata list order
2. When loading: tensor[i] corresponds to metadata[i]["qid"]
3. Use qid from metadata for cross-referencing across personas or with original dataset

File structure per persona:
    activations.safetensors  # Keys: "question_0000", "question_0001", ...
    metadata.json            # {"questions": [...]}
"""

# HACK: The current storing structure is a bit hacky and we can decide on what would be the best approach
# Currently it works but will be discussed I wouldn't rely too much on it.
# The idea is to keep things similar the things that will likely change though is the metadata

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def _model_dir_name(model_name: str) -> str:
    return model_name.replace("/", "__")


def _tensor_key(index: int) -> str:
    return f"question_{index:04d}"


def build_activation_path(
    root_dir: str | Path,
    model_name: str,
    prompt_variant: str,
    persona_id: str,
) -> Path:
    return Path(root_dir) / _model_dir_name(model_name) / prompt_variant / persona_id


def save_per_question_activations(
    root_dir: str | Path,
    model_name: str,
    prompt_variant: str,
    persona_id: str,
    per_question_activations: list[torch.Tensor],
    metadata: list[dict],
) -> Path:
    """Save per-question activation tensors and rich metadata.

    NOTE: Order matters! Tensor at index i maps to metadata[i]. The metadata
    list should contain a "qid" key for each question to enable cross-referencing
    with the original dataset (SynthPersonaDataset.get_qa).

    Args:
        per_question_activations: List of activation tensors, one per question.
            The order of this list MUST match the order of metadata.
        metadata: List of dicts with keys: qid, question, answer, seq_len,
            answer_start, answer_end. The order MUST match per_question_activations.
    """
    n_questions = len(per_question_activations)
    if len(metadata) != n_questions:
        raise ValueError("metadata must have same length as per_question_activations")

    tensors: dict[str, torch.Tensor] = {}
    for index, activation in enumerate(per_question_activations):
        tensors[_tensor_key(index)] = activation.detach().cpu()

    artifact_dir = build_activation_path(
        root_dir=root_dir,
        model_name=model_name,
        prompt_variant=prompt_variant,
        persona_id=persona_id,
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)

    tensor_path = artifact_dir / "activations.safetensors"
    save_file(tensors, str(tensor_path))

    metadata_path = artifact_dir / "metadata.json"
    metadata_path.write_text(json.dumps({"questions": metadata}, indent=2))

    return artifact_dir


def load_per_question_activations(
    root_dir: str | Path,
    model_name: str,
    prompt_variant: str,
    persona_id: str,
) -> tuple[list[torch.Tensor], list[dict]]:
    """Load activation tensors and their metadata.

    NOTE: The returned lists are aligned by index. Use metadata[i]["qid"]
    to cross-reference with the original dataset (SynthPersonaDataset.get_qa).

    Returns:
        per_question_activations: List of activation tensors, ordered by index.
        metadata: List of metadata dicts with "qid" key. aligned with tensors.
            Use metadata[i]["qid"] to match with SynthPersonaDataset.get_qa(persona_id).
    """
    artifact_dir = build_activation_path(
        root_dir=root_dir,
        model_name=model_name,
        prompt_variant=prompt_variant,
        persona_id=persona_id,
    )

    tensors = load_file(str(artifact_dir / "activations.safetensors"))
    metadata = json.loads((artifact_dir / "metadata.json").read_text())

    questions = metadata.get("questions", metadata.get("qids", []))
    if (
        isinstance(questions, list)
        and len(questions) > 0
        and isinstance(questions[0], str)
    ):
        questions = [{"qid": qid} for qid in questions]

    n_questions = len(questions)
    per_question_activations = [
        tensors[_tensor_key(index)] for index in range(n_questions)
    ]

    return per_question_activations, questions
