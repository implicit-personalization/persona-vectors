import json
import os
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

SUPPORTED_VARIANTS: tuple[str, ...] = ("templated", "biography")


def model_dir_name(model_name: str) -> str:
    return model_name.replace("/", "__")


def _artifact_path(
    root_dir: str | Path,
    model_name: str,
    prompt_variant: str,
    persona_id: str,
) -> Path:
    return Path(root_dir) / model_dir_name(model_name) / prompt_variant / persona_id


class ActivationStore:
    """Artifact storage for per-question activation vectors under a root directory."""

    def __init__(self, model_name: str, root_dir: str | Path | None = None) -> None:
        self.model_name = model_name
        self.root_dir = (
            Path(root_dir)
            if root_dir is not None
            else Path(os.environ.get("ARTIFACTS_DIR", "artifacts")) / "activations"
        )

    def _path(self, prompt_variant: str, persona_id: str) -> Path:
        return _artifact_path(
            self.root_dir, self.model_name, prompt_variant, persona_id
        )

    def save(
        self,
        prompt_variant: str,
        persona_id: str,
        persona_name: str,
        per_question_vectors: torch.Tensor,
        questions: list[str],
    ) -> Path:
        """Save per-question activation vectors and metadata. Returns the artifact directory."""
        if per_question_vectors.ndim != 3:
            raise ValueError(
                "per_question_vectors must have shape (n_questions, num_layers, hidden_size)"
            )
        if len(questions) != per_question_vectors.shape[0]:
            raise ValueError("number of questions must match first tensor dimension")

        artifact_dir = self._path(prompt_variant, persona_id)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        save_file(
            {"per_question_vectors": per_question_vectors.detach().cpu()},
            str(artifact_dir / "activations.safetensors"),
        )
        (artifact_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "persona_id": persona_id,
                    "persona_name": persona_name,
                    "questions": questions,
                },
                indent=2,
            )
        )
        return artifact_dir

    def load(
        self,
        prompt_variant: str,
        persona_id: str,
    ) -> tuple[torch.Tensor, list[str]]:
        """Load per-question vectors and questions. Returns (vectors, questions)."""
        artifact_dir = self._path(prompt_variant, persona_id)
        tensors = load_file(str(artifact_dir / "activations.safetensors"))
        metadata = json.loads((artifact_dir / "metadata.json").read_text())
        return tensors["per_question_vectors"], metadata["questions"]


def list_personas(
    root_dir: str | Path,
    model_name: str,
    variants: list[str],
) -> list[str]:
    """List persona ids available for every requested variant."""

    root = Path(root_dir)
    shared_personas: set[str] | None = None

    for variant in variants:
        model_dir = root / model_dir_name(model_name) / variant
        if not model_dir.exists():
            return []

        variant_personas = {d.name for d in model_dir.iterdir() if d.is_dir()}
        shared_personas = (
            variant_personas
            if shared_personas is None
            else shared_personas & variant_personas
        )
        if not shared_personas:
            return []

    return sorted(shared_personas or set())


def load_persona_names(
    root_dir: str | Path,
    model_name: str,
    variants: list[str],
    persona_ids: list[str],
) -> dict[str, str]:
    """Load display names from saved activation metadata."""

    names: dict[str, str] = {}
    for persona_id in persona_ids:
        for variant in variants:
            metadata_path = (
                _artifact_path(root_dir, model_name, variant, persona_id)
                / "metadata.json"
            )
            try:
                metadata = json.loads(metadata_path.read_text())
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                continue

            persona_name = metadata.get("persona_name")
            if isinstance(persona_name, str) and persona_name:
                names[persona_id] = persona_name
                break

    return names


def list_layers(
    root_dir: str | Path,
    model_name: str,
    variants: list[str],
    persona_ids: list[str],
) -> list[int]:
    """List layer indices shared by all matching saved activation files."""

    store = ActivationStore(model_name, root_dir)
    shared_layers: set[int] | None = None

    for variant in variants:
        for persona_id in persona_ids:
            try:
                vectors, _ = store.load(variant, persona_id)
            except (FileNotFoundError, KeyError, OSError, ValueError):
                continue

            layers = set(range(vectors.shape[1]))
            shared_layers = layers if shared_layers is None else shared_layers & layers

    return sorted(shared_layers or set())


def load_mean_activations(
    root_dir: str | Path,
    model_name: str,
    persona_ids: list[str],
    variant_a: str,
    variant_b: str,
) -> tuple[list[tuple[str, torch.Tensor, torch.Tensor]], dict[str, str], list[str]]:
    """Load per-persona mean activation vectors for two variants."""

    store = ActivationStore(model_name, root_dir)
    persona_names = load_persona_names(
        root_dir, model_name, [variant_a, variant_b], persona_ids
    )
    traces: list[tuple[str, torch.Tensor, torch.Tensor]] = []
    errors: list[str] = []

    for persona_id in persona_ids:
        try:
            vectors_a, _ = store.load(variant_a, persona_id)
            vectors_b, _ = store.load(variant_b, persona_id)
        except (FileNotFoundError, KeyError, OSError, ValueError) as exc:
            errors.append(f"{persona_id}: {exc}")
            continue

        traces.append(
            (
                persona_id,
                vectors_a.float().mean(dim=0),
                vectors_b.float().mean(dim=0),
            )
        )

    return traces, persona_names, errors
