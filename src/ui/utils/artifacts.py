from collections.abc import Callable
from pathlib import Path

import torch

from src.activation_io import load_activation_metadata, load_per_question_vectors


def model_dir_name(model_name: str) -> str:
    """Encode a model name for use in artifact paths."""

    return model_name.replace("/", "__")


def list_available_personas(
    artifacts_root: str | Path,
    model_name: str,
    variants: list[str],
) -> list[str]:
    """List persona ids with saved activations for the given model and variants."""

    persona_ids: set[str] = set()
    root = Path(artifacts_root)
    for variant in variants:
        model_dir = root / model_dir_name(model_name) / variant
        if not model_dir.exists():
            continue
        persona_ids.update(d.name for d in model_dir.iterdir() if d.is_dir())
    return sorted(persona_ids)


def load_persona_names(
    artifacts_root: str | Path,
    model_name: str,
    variants: list[str],
    persona_ids: list[str],
) -> dict[str, str]:
    """Load display names from saved activation metadata."""

    names: dict[str, str] = {}
    for persona_id in persona_ids:
        for variant in variants:
            try:
                metadata = load_activation_metadata(
                    root_dir=artifacts_root,
                    model_name=model_name,
                    prompt_variant=variant,
                    persona_id=persona_id,
                )
            except Exception:
                continue

            persona_name = metadata.get("persona_name")
            if isinstance(persona_name, str) and persona_name:
                names[persona_id] = persona_name
                break

    return names


def artifact_persona_options(
    artifacts_root: str | Path,
    model_name: str,
    variants: list[str],
) -> tuple[list[str], dict[str, str]]:
    """Return persona ids and names for the selected artifacts."""

    persona_options = list_available_personas(artifacts_root, model_name, variants)
    persona_names = load_persona_names(
        artifacts_root,
        model_name,
        variants,
        persona_options,
    )
    return persona_options, persona_names


def list_available_layers(
    artifacts_root: str | Path,
    model_name: str,
    variants: list[str],
    persona_ids: list[str],
) -> list[int]:
    """List layer indices present in any matching saved activation file."""

    layers: set[int] = set()
    for variant in variants:
        for persona_id in persona_ids:
            try:
                vectors, _ = load_per_question_vectors(
                    root_dir=artifacts_root,
                    model_name=model_name,
                    prompt_variant=variant,
                    persona_id=persona_id,
                )
            except Exception:
                continue
            layers.update(range(vectors.shape[1]))
    return sorted(layers)


def load_cosine_traces(
    artifacts_root: str | Path,
    model_name: str,
    persona_ids: list[str],
    variant_a: str,
    variant_b: str,
) -> tuple[list[tuple[str, torch.Tensor, torch.Tensor]], dict[str, str], list[str]]:
    """Load mean activation traces for pairwise cosine-similarity plots."""

    persona_names = load_persona_names(
        artifacts_root,
        model_name,
        [variant_a, variant_b],
        persona_ids,
    )
    traces: list[tuple[str, torch.Tensor, torch.Tensor]] = []
    errors: list[str] = []

    for persona_id in persona_ids:
        try:
            vectors_a, _ = load_per_question_vectors(
                root_dir=artifacts_root,
                model_name=model_name,
                prompt_variant=variant_a,
                persona_id=persona_id,
            )
            vectors_b, _ = load_per_question_vectors(
                root_dir=artifacts_root,
                model_name=model_name,
                prompt_variant=variant_b,
                persona_id=persona_id,
            )
        except Exception as exc:
            errors.append(f"{persona_id}: {exc}")
            continue

        traces.append(
            (persona_id, vectors_a.float().mean(dim=0), vectors_b.float().mean(dim=0))
        )

    return traces, persona_names, errors


def load_embedding_samples(
    artifacts_root: str | Path,
    model_name: str,
    persona_ids: list[str],
    variant: str,
    selected_layers: list[int],
    project_fn: Callable[[torch.Tensor], torch.Tensor],
    persona_names: dict[str, str],
) -> tuple[list[tuple[int, torch.Tensor, list[str], list[str]]], list[str]]:
    """Load samples for 2D projections without re-reading each layer from disk."""

    plots: list[tuple[int, torch.Tensor, list[str], list[str]]] = []
    errors: list[str] = []
    vectors_by_persona: dict[str, torch.Tensor] = {}

    for persona_id in persona_ids:
        try:
            vectors, _ = load_per_question_vectors(
                root_dir=artifacts_root,
                model_name=model_name,
                prompt_variant=variant,
                persona_id=persona_id,
            )
        except Exception as exc:
            errors.append(f"{persona_id} / {variant}: {exc}")
            continue

        vectors_by_persona[persona_id] = vectors

    for layer_idx in selected_layers:
        samples: list[torch.Tensor] = []
        labels: list[str] = []
        hover_text: list[str] = []

        for persona_id, vectors in vectors_by_persona.items():
            layer_vectors = vectors[:, int(layer_idx), :]
            samples.append(layer_vectors)
            labels.extend([persona_id] * layer_vectors.shape[0])
            display_name = persona_names.get(persona_id) or persona_id
            hover_text.extend(
                [
                    f"<b>{display_name}</b><br>{variant}",
                ]
                * layer_vectors.shape[0]
            )

        if not samples:
            continue

        all_samples = torch.cat(samples, dim=0)
        if all_samples.shape[0] < 2:
            errors.append(
                f"Layer {layer_idx}: need at least 2 samples for {variant} analysis"
            )
            continue

        try:
            coords = project_fn(all_samples)
        except Exception as exc:
            errors.append(f"Layer {layer_idx}: {exc}")
            continue
        plots.append((int(layer_idx), coords, labels, hover_text))

    return plots, errors
