from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

from persona_vectors.artifacts import ActivationStore, list_personas, load_persona_names


@dataclass(frozen=True)
class LayeredSamples:
    """Samples ready for per-layer PCA or similarity."""

    vectors: torch.Tensor
    labels: list[str]
    hover_text: list[str]


def load_persona_mean_samples(
    root_dir: str | Path,
    model_name: str,
    variant: str,
    mask_strategy: object,
    persona_ids: list[str] | None = None,
) -> LayeredSamples:
    """Load one mean activation sample per persona.

    Args:
        root_dir: Activation artifact root, usually ``artifacts/activations``.
        model_name: HuggingFace model id used at extraction time.
        variant: Prompt variant to load.
        mask_strategy: Saved mask strategy to load.
        persona_ids: Optional subset. If omitted, all personas present for the
            variant/mask strategy are loaded.
    """

    if persona_ids is None:
        persona_ids = list_personas(
            root_dir, model_name, [variant], mask_strategy=mask_strategy
        )
    if not persona_ids:
        raise FileNotFoundError(
            f"No personas found for {model_name!r} / {variant!r} / {mask_strategy!r}"
        )

    store = ActivationStore(model_name, root_dir=root_dir)
    persona_names = load_persona_names(
        root_dir, model_name, [variant], persona_ids, mask_strategy=mask_strategy
    )

    vectors: list[torch.Tensor] = []
    labels: list[str] = []
    hover_text: list[str] = []

    for persona_id in persona_ids:
        acts, _ = store.load(variant, persona_id, mask_strategy=mask_strategy)
        if acts.ndim != 3:
            raise ValueError(
                f"Expected {persona_id!r} activations to have shape (n_samples, n_layers, hidden)"
            )
        name = persona_names.get(persona_id, persona_id)
        vectors.append(acts.float().mean(dim=0))
        labels.append(name)
        hover_text.append(
            f"Persona: {name}<br>ID: {persona_id}<br>Questions averaged: {acts.shape[0]}"
        )

    return LayeredSamples(torch.stack(vectors), labels, hover_text)


def _center_features(samples: torch.Tensor) -> torch.Tensor:
    return samples.float() - samples.float().mean(dim=0, keepdim=True)


def pairwise_cosine_similarity(
    vectors: list[torch.Tensor], center: bool = False
) -> torch.Tensor:
    """Compute pairwise cosine similarity between vectors.

    Args:
        vectors: List of 1-D tensors (one per persona/condition).
        center: If True, subtract the mean vector across the list before
            normalising. LLM residual-stream means share a large DC component
            that pushes every pairwise cosine toward ~1; centering removes it
            so the remaining structure (which personas actually cluster) shows.
    """

    if not vectors:
        raise ValueError("vectors must not be empty")
    if any(vector.ndim != 1 for vector in vectors):
        raise ValueError("vectors must be 1-D tensors")

    stacked = torch.stack([vector.float() for vector in vectors])
    return cosine_similarity_matrix(stacked, center=center)


def cosine_similarity_matrix(
    samples: torch.Tensor, center: bool = True
) -> torch.Tensor:
    """Cosine similarity for a 2-D sample matrix, centered by default."""

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


def project_pca_centered(samples: torch.Tensor) -> torch.Tensor:
    """Project samples to 2D PCA after explicit feature centering."""

    return project_pca(_center_features(samples))


def run_saved_activation_analysis(
    model_name: str,
    activations_dir: str | Path,
    output_dir: str | Path,
    variant: str,
    mask_strategy: object,
    persona_ids: list[str] | None = None,
    layers: list[int] | None = None,
) -> dict[str, Path]:
    """Create interactive PCA and similarity HTML files from saved activations."""

    from persona_vectors.plots import build_layered_figure

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    samples = load_persona_mean_samples(
        activations_dir,
        model_name,
        variant,
        mask_strategy=mask_strategy,
        persona_ids=persona_ids,
    )
    figure_specs = [("persona_mean", "pca"), ("persona_mean", "similarity")]
    title_suffix = {
        "pca": "centered PCA",
        "similarity": "centered cosine similarity",
    }

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
    return outputs


def project_umap(samples: torch.Tensor) -> torch.Tensor:
    """Project samples to 2D using UMAP.

    Args:
        samples: Tensor with shape (n_samples, hidden_size).

    Returns:
        Tensor with shape (n_samples, 2).
    """
    if samples.ndim != 2:
        raise ValueError("samples must have shape (n_samples, hidden_size)")

    try:
        import umap
    except ImportError as exc:
        raise ImportError("umap-learn is required for UMAP projections") from exc

    embedding = umap.UMAP(n_components=2, random_state=1337).fit_transform(
        samples.float().cpu().numpy()
    )
    return torch.from_numpy(embedding)


def project_umap_centered(samples: torch.Tensor) -> torch.Tensor:
    """Project samples to 2D UMAP after explicit feature centering."""

    return project_umap(_center_features(samples))


def pca_explained_variance(
    samples: torch.Tensor, n_components: int | None = None
) -> np.ndarray:
    """Return the explained variance ratio for each principal component."""

    if samples.ndim != 2:
        raise ValueError("samples must have shape (n_samples, hidden_size)")

    X = samples.float().cpu().numpy()
    max_components = min(X.shape)
    if n_components is None:
        n_components = max_components
    else:
        n_components = min(n_components, max_components)

    pca = PCA(n_components=n_components).fit(X)
    return pca.explained_variance_ratio_
