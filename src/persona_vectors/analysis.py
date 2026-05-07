from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from persona_data.synth_persona import BASELINE_PERSONA_ID
from sklearn.decomposition import PCA

from persona_vectors.artifacts import ActivationStore, HFActivationStore

PersonaVectorStore = ActivationStore | HFActivationStore


@dataclass(frozen=True)
class LayeredSamples:
    """Samples ready for per-layer PCA or similarity."""

    vectors: torch.Tensor
    labels: list[str]
    hover_text: list[str]


def list_comparison_personas(
    store: PersonaVectorStore,
    variants: list[str] | tuple[str, ...],
    mask_strategy: object | None = None,
    *,
    include_baseline: bool = False,
) -> list[str]:
    """Return shared persona ids for comparisons, with baseline filtering."""

    persona_ids = store.list_personas(variants, mask_strategy=mask_strategy)
    if include_baseline:
        return persona_ids
    return [
        persona_id for persona_id in persona_ids if persona_id != BASELINE_PERSONA_ID
    ]


def _resolve_personas(
    store: PersonaVectorStore,
    variants: list[str],
    mask_strategy: object | None,
    persona_ids: list[str] | None,
) -> list[str]:
    if persona_ids is None:
        persona_ids = store.list_personas(variants, mask_strategy=mask_strategy)
    if not persona_ids:
        raise FileNotFoundError(
            f"No personas found for {store.model_name!r} / {variants!r} / {mask_strategy!r}"
        )
    return persona_ids


def _load_variant_samples(
    store: PersonaVectorStore,
    variant: str,
    mask_strategy: object | None,
    persona_ids: list[str],
) -> LayeredSamples:
    """One persona vector per persona for a single variant."""
    persona_names = store.persona_names(
        persona_ids, variants=[variant], mask_strategy=mask_strategy
    )
    vectors, labels, hover_text = [], [], []
    for persona_id in persona_ids:
        acts = store.load(variant, persona_id, mask_strategy=mask_strategy)
        name = persona_names.get(persona_id, persona_id)
        vectors.append(acts.float())
        labels.append(name)
        hover_text.append(f"Persona: {name}<br>ID: {persona_id}")
    return LayeredSamples(torch.stack(vectors), labels, hover_text)


def load_persona_vectors(
    store: PersonaVectorStore,
    variant: str,
    mask_strategy: object | None = None,
    persona_ids: list[str] | None = None,
) -> LayeredSamples:
    """Load saved persona vectors for a single variant.

    Each vector is a ``(num_layers, hidden_size)`` tensor. Extraction has
    already averaged across QA pairs and masked tokens.
    """
    persona_ids = _resolve_personas(store, [variant], mask_strategy, persona_ids)
    return _load_variant_samples(store, variant, mask_strategy, persona_ids)


def load_variant_vectors(
    store: PersonaVectorStore,
    variants: list[str] | tuple[str, ...],
    mask_strategy: object | None = None,
    persona_ids: list[str] | None = None,
) -> dict[str, LayeredSamples]:
    """Load saved persona vectors for multiple variants in a shared order.

    Returns a dict mapping variant name to a ``LayeredSamples`` where each
    entry is one ``(num_layers, hidden_size)`` tensor per persona.
    """
    requested_variants = list(variants)
    if not requested_variants:
        raise ValueError("At least one variant is required")
    persona_ids = _resolve_personas(
        store, requested_variants, mask_strategy, persona_ids
    )
    return {
        variant: _load_variant_samples(store, variant, mask_strategy, persona_ids)
        for variant in requested_variants
    }


def _center_features(samples: torch.Tensor) -> torch.Tensor:
    return samples.float() - samples.float().mean(dim=0, keepdim=True)


def cosine_similarity_matrix(
    samples: torch.Tensor, center: bool = True
) -> torch.Tensor:
    """Cosine similarity for a 2-D sample matrix, centered by default.

    Centering subtracts the per-feature mean across rows before normalising.
    LLM residual-stream means share a large DC component that pushes every raw
    pairwise cosine toward ~1; centering removes it so the remaining persona
    cluster structure shows.
    """

    if samples.ndim != 2:
        raise ValueError("samples must have shape (n_samples, hidden_size)")
    if center:
        samples = _center_features(samples)
    normalized = F.normalize(samples.float(), dim=1)
    return normalized @ normalized.T


def project_pca(samples: torch.Tensor) -> torch.Tensor:
    """Project samples to 2D using PCA.

    Args:
        samples: Tensor with shape (n_samples, hidden_size).

    Returns:
        Tensor with shape (n_samples, 2).
    """
    if samples.ndim != 2:
        raise ValueError("samples must have shape (n_samples, hidden_size)")

    embedding = PCA(n_components=2).fit_transform(samples.float().cpu().numpy())
    return torch.from_numpy(embedding)


def pca_explained_variance(
    samples: torch.Tensor, n_components: int | None = None
) -> np.ndarray:
    """Return the explained variance ratio for each principal component."""

    if samples.ndim != 2:
        raise ValueError("samples must have shape (n_samples, hidden_size)")

    x = samples.float().cpu().numpy()
    max_components = min(x.shape)
    if n_components is None:
        n_components = max_components
    else:
        n_components = min(n_components, max_components)

    pca = PCA(n_components=n_components).fit(x)
    return pca.explained_variance_ratio_


def _scree_layers(num_layers: int, layers: list[int] | None) -> list[int]:
    """Use requested layers, or a small representative set for compact plots."""
    if layers is not None:
        selected = list(layers)
    elif num_layers <= 4:
        selected = list(range(num_layers))
    else:
        selected = sorted({0, num_layers // 3, (2 * num_layers) // 3, num_layers - 1})

    invalid = [layer for layer in selected if layer < 0 or layer >= num_layers]
    if invalid:
        raise ValueError(
            f"Invalid layer(s) for tensor with {num_layers} layers: {invalid}"
        )
    return selected


def run_saved_activation_analysis(
    model_name: str,
    activations_dir: str | Path,
    output_dir: str | Path,
    variant: str,
    mask_strategy: object,
    persona_ids: list[str] | None = None,
    include_baseline: bool = False,
    layers: list[int] | None = None,
) -> dict[str, Path]:
    """Create interactive PCA and similarity HTML files from saved activations."""

    from persona_vectors.plots import (
        build_layered_figure,
        build_pair_similarity_figure,
        plot_scree,
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    store = ActivationStore(
        model_name,
        root_dir=activations_dir,
        mask_strategy=mask_strategy,
    )

    if persona_ids is None:
        persona_ids = list_comparison_personas(
            store,
            [variant],
            include_baseline=include_baseline,
        )
    samples = load_persona_vectors(store, variant, persona_ids=persona_ids)
    figure_specs = [("persona_vector", "pca"), ("persona_vector", "similarity")]
    title_suffix = {"pca": "PCA", "similarity": "centered cosine similarity"}
    outputs: dict[str, Path] = {}
    for name, kind in figure_specs:
        fig = build_layered_figure(
            samples,
            kind,
            layers=layers,
            title=f"{variant} {mask_strategy} {name} {title_suffix[kind]}",
        )
        path = output / f"{variant}_{mask_strategy}_{name}_{kind}.html"
        fig.write_html(str(path))
        outputs[f"{name}_{kind}"] = path

    pair_fig = build_pair_similarity_figure(
        samples,
        layers=layers,
        title=f"{variant} {mask_strategy} persona-pair similarity across layers",
    )
    pair_path = output / f"{variant}_{mask_strategy}_persona_pair_similarity.html"
    pair_fig.write_html(str(pair_path))
    outputs["persona_pair_similarity"] = pair_path

    scree_layers = _scree_layers(int(samples.vectors.shape[1]), layers)
    scree_fig = plot_scree(
        {
            f"layer {layer}": pca_explained_variance(samples.vectors[:, layer, :])
            for layer in scree_layers
        },
        title=f"{variant} {mask_strategy} PCA explained variance",
        show=False,
    )
    scree_path = output / f"{variant}_{mask_strategy}_pca_scree.html"
    scree_fig.write_html(str(scree_path))
    outputs["pca_scree"] = scree_path
    return outputs


def project_umap(samples: torch.Tensor) -> torch.Tensor:
    """Project samples to 2D using UMAP after centering features.

    Centering removes the shared DC component before UMAP fits, matching the
    convention used by the centered cosine views.
    """
    if samples.ndim != 2:
        raise ValueError("samples must have shape (n_samples, hidden_size)")

    try:
        import umap
    except ImportError as exc:
        raise ImportError("umap-learn is required for UMAP projections") from exc

    centered = _center_features(samples)
    embedding = umap.UMAP(n_components=2, random_state=1337).fit_transform(
        centered.float().cpu().numpy()
    )
    return torch.from_numpy(embedding)
