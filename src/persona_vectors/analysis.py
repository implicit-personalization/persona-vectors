from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.neighbors import kneighbors_graph

from persona_vectors.artifacts import (
    PersonaVectorSource,
    PersonaVectorStore,
)


@dataclass(frozen=True)
class LayeredSamples:
    """Samples ready for per-layer PCA or similarity."""

    vectors: torch.Tensor
    labels: list[str]
    hover_text: list[str]


def _resolve_personas(
    store: PersonaVectorSource,
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
    store: PersonaVectorSource,
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
    store: PersonaVectorSource,
    variant: str,
    mask_strategy: object | None = None,
    persona_ids: list[str] | None = None,
) -> LayeredSamples:
    """Load saved persona vectors for a single variant (one ``(num_layers, hidden_size)`` tensor per persona)."""
    persona_ids = _resolve_personas(store, [variant], mask_strategy, persona_ids)
    return _load_variant_samples(store, variant, mask_strategy, persona_ids)


def load_variant_vectors(
    store: PersonaVectorSource,
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
    x = samples.float()
    return x - x.mean(dim=0, keepdim=True)


def _validate_projection(
    samples: torch.Tensor,
    n_components: int,
    *,
    method: str,
    min_samples: int = 2,
) -> None:
    if samples.ndim != 2:
        raise ValueError("samples must have shape (n_samples, hidden_size)")
    if n_components < 1:
        raise ValueError("n_components must be at least 1")

    n_samples, n_features = samples.shape
    if n_samples < min_samples:
        raise ValueError(
            f"{method} requires at least {min_samples} samples; got {n_samples}"
        )
    max_components = min(n_samples, n_features)
    if n_components > max_components:
        raise ValueError(
            f"{method} n_components={n_components} must be <= min(n_samples, hidden_size)={max_components}"
        )


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


def project_pca(samples: torch.Tensor, n_components: int = 2) -> torch.Tensor:
    """Project a ``(n_samples, hidden_size)`` tensor to ``n_components`` PCA dimensions."""
    _validate_projection(samples, n_components, method="PCA")

    embedding = PCA(n_components=n_components).fit_transform(
        samples.float().cpu().numpy()
    )
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
    store = PersonaVectorStore(
        model_name,
        root_dir=activations_dir,
        mask_strategy=mask_strategy,
    )

    if persona_ids is None:
        persona_ids = store.list_personas(
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


def prepare_cluster_samples(
    samples: torch.Tensor, *, center: bool = True, normalize: bool = True
) -> torch.Tensor:
    """Preprocess a 2-D sample matrix for distance-based clustering.

    Centering removes the shared residual-stream component across personas.
    L2 normalization keeps clusters focused on direction/profile similarity
    instead of raw vector magnitude.
    """

    if samples.ndim != 2:
        raise ValueError("samples must have shape (n_samples, hidden_size)")
    prepared = samples.float()
    if center:
        prepared = _center_features(prepared)
    if normalize:
        prepared = F.normalize(prepared, dim=1)
    return prepared


def prepare_layer_mean_cluster_samples(
    vectors: torch.Tensor, *, center: bool = True, normalize: bool = True
) -> torch.Tensor:
    """Return one stable clustering vector per persona from layered vectors.

    For ``(n_samples, n_layers, hidden_size)`` inputs, preprocessing is applied
    within each layer before averaging across layers. That prevents larger-norm
    layers from dominating the global cluster labels used to color layer sliders.
    """

    if vectors.ndim != 3:
        raise ValueError("vectors must have shape (n_samples, n_layers, hidden_size)")
    prepared = vectors.float()
    if center:
        prepared = prepared - prepared.mean(dim=0, keepdim=True)
    if normalize:
        prepared = F.normalize(prepared, dim=2)
    return prepared.mean(dim=1)


def _samples_to_numpy(
    samples: torch.Tensor, *, center: bool = True, normalize: bool = True
) -> np.ndarray:
    return (
        prepare_cluster_samples(samples, center=center, normalize=normalize)
        .cpu()
        .numpy()
    )


def cluster_kmeans(
    samples: torch.Tensor,
    n_clusters: int,
    *,
    seed: int = 0,
    center: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    """K-means (k-means++ init) cluster labels for a (n_samples, hidden) tensor."""
    return KMeans(n_clusters=n_clusters, n_init="auto", random_state=seed).fit_predict(
        _samples_to_numpy(samples, center=center, normalize=normalize)
    )


def _cosine_affinity(samples: torch.Tensor, *, center: bool) -> np.ndarray:
    """Nonnegative precomputed affinity from centered cosine similarity.

    Cosine similarity lies in [-1, 1]; spectral clustering and the graph
    Laplacian need nonnegative weights, so map it to [0, 1].
    """
    sim = cosine_similarity_matrix(samples, center=center).cpu().numpy()
    return (sim + 1.0) / 2.0


def cluster_spectral(
    samples: torch.Tensor,
    n_clusters: int,
    *,
    affinity: str = "nearest_neighbors",
    n_neighbors: int = 8,
    seed: int = 0,
    center: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    """Spectral clustering labels for a (n_samples, hidden) tensor.

    Clusters on the affinity graph rather than in Euclidean space, so it
    recovers non-convex persona structure that k-means misses. ``affinity``
    is either ``"nearest_neighbors"`` (kNN graph over centered/unit vectors,
    matching the Isomap convention) or ``"cosine"`` (the repo's centered
    cosine similarity as a precomputed affinity).
    """
    n_samples = int(samples.shape[0])
    if affinity == "nearest_neighbors":
        return SpectralClustering(
            n_clusters=n_clusters,
            affinity="nearest_neighbors",
            n_neighbors=min(n_neighbors, n_samples - 1),
            assign_labels="kmeans",
            random_state=seed,
        ).fit_predict(_samples_to_numpy(samples, center=center, normalize=normalize))
    if affinity == "cosine":
        return SpectralClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            assign_labels="kmeans",
            random_state=seed,
        ).fit_predict(_cosine_affinity(samples, center=center))
    raise ValueError("affinity must be one of: nearest_neighbors, cosine")


def laplacian_eigenvalues(
    samples: torch.Tensor,
    *,
    affinity: str = "nearest_neighbors",
    n_neighbors: int = 8,
    center: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    """Ascending eigenvalues of the symmetric normalized graph Laplacian.

    The largest gap between successive eigenvalues (the eigengap) is the
    spectral analogue of a scree elbow and suggests the natural cluster count.
    """
    from scipy.sparse.csgraph import laplacian

    n_samples = int(samples.shape[0])
    if affinity == "nearest_neighbors":
        graph = kneighbors_graph(
            _samples_to_numpy(samples, center=center, normalize=normalize),
            n_neighbors=min(n_neighbors, n_samples - 1),
            include_self=False,
        ).toarray()
        weights = np.maximum(graph, graph.T)
    elif affinity == "cosine":
        weights = _cosine_affinity(samples, center=center)
    else:
        raise ValueError("affinity must be one of: nearest_neighbors, cosine")

    lap = laplacian(weights, normed=True)
    return np.sort(np.linalg.eigvalsh(lap))


def project_isomap(
    samples: torch.Tensor, n_components: int = 2, *, n_neighbors: int = 8
) -> torch.Tensor:
    """Project samples with Isomap over centered/unit persona vectors.

    Isomap is useful as a geometry check against PCA and UMAP: it preserves
    shortest-path distances on a nearest-neighbor graph, so stable structure in
    this view suggests a low-dimensional geodesic organization rather than only
    linear variance or UMAP-specific local packing.
    """

    _validate_projection(samples, n_components, method="Isomap", min_samples=3)
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be at least 1")

    n_neighbors = min(n_neighbors, int(samples.shape[0]) - 1)
    embedding = Isomap(
        n_components=n_components,
        n_neighbors=n_neighbors,
        metric="euclidean",
    ).fit_transform(_samples_to_numpy(samples, center=True, normalize=True))
    return torch.from_numpy(embedding)


def project_umap(samples: torch.Tensor, n_components: int = 2) -> torch.Tensor:
    """Project samples to ``n_components`` dimensions using UMAP after centering features.

    Centering removes the shared DC component before UMAP fits, matching the
    convention used by the centered cosine views.
    """
    _validate_projection(samples, n_components, method="UMAP", min_samples=3)

    try:
        import umap
    except ImportError as exc:
        raise ImportError("umap-learn is required for UMAP projections") from exc

    centered = _center_features(samples)
    n_neighbors = min(15, int(samples.shape[0]) - 1)
    embedding = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        init="random",
        random_state=42,
        n_jobs=1,
    ).fit_transform(centered.float().cpu().numpy())
    return torch.from_numpy(embedding)
