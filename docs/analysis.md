# Analysis

Analysis operates on saved persona vectors. A saved vector has shape `(num_layers, hidden_size)`; a loaded sample collection has shape `(n_personas, num_layers, hidden_size)`.

Core modules:

- `src/persona_vectors/analysis.py`
- `src/persona_vectors/plots/` (package; probe-specific plots live in `plots/probes.py`)

## CLI

```bash
uv run python main.py analyze \
  --model google/gemma-2-9b-it \
  --variant biography \
  --mask-strategy answer_mean \
  --out artifacts/plots
```

This writes interactive HTML files for PCA, centered cosine similarity, persona-pair similarity, and PCA scree curves.

## Loading Vectors

```python
from persona_vectors.analysis import load_persona_vectors, load_variant_vectors
from persona_vectors.artifacts import PersonaVectorStore

store = PersonaVectorStore("google/gemma-2-9b-it", mask_strategy="answer_mean")

persona_ids = store.list_personas(["biography"])
samples = load_persona_vectors(store, "biography", persona_ids=persona_ids)
# keeps the persona order shared across variants.
by_variant = load_variant_vectors(store, ["biography", "templated"])
```


## Similarity and Projection

```python
from persona_vectors.analysis import (
    cosine_similarity_matrix,
    pca_explained_variance,
    project_isomap,
    project_pca,
    project_umap,
)

layer_vectors = samples.vectors[:, 20, :]
similarity = cosine_similarity_matrix(layer_vectors)
pca = project_pca(layer_vectors, n_components=2)
isomap = project_isomap(layer_vectors, n_components=2)
variance = pca_explained_variance(layer_vectors)
```

Cosine similarity is centered by default — removes the shared residual-stream DC component that otherwise pushes raw cosines toward 1.

## Clustering

```python
from persona_vectors.analysis import (
    cluster_kmeans,
    cluster_spectral,
    laplacian_eigenvalues,
    prepare_layer_mean_cluster_samples,
)

cluster_input = prepare_layer_mean_cluster_samples(samples.vectors)
labels = cluster_kmeans(cluster_input, n_clusters=5, center=False, normalize=False)
spectral_labels = cluster_spectral(samples.vectors[:, 20, :], n_clusters=5)
eigenvalues = laplacian_eigenvalues(samples.vectors[:, 20, :])
```

`cluster_kmeans(samples, n_clusters)` returns k-means labels.
`cluster_spectral(samples, n_clusters)` clusters an affinity graph; use
`laplacian_eigenvalues(...)` as an eigengap diagnostic when choosing `k`.

Clustering inputs are centered and L2-normalized by default.

For projection cluster colors, pass `n_clusters=k` to
`build_layered_figure(...)`, or call `prepare_kmeans_groups(...)` once and
reuse as `groups=...`. `cluster_method="kmeans"` is the default;
`cluster_method="spectral"` uses spectral clustering. `cluster_mode` controls
stability across layers:

- `mean_across_layers` — colors from the centered/unit mean vector per persona.
- `first_layer` — colors from the first plotted layer.
- `per_layer` — recompute every frame.

Pass explicit categorical labels as `groups=...`, or numeric/ordinal values as `color_values=...`. Dendrograms use `plot_persona_dendrogram(..., linkage=...)`.

## Plot Helpers

All plot helpers return a Plotly `go.Figure`.

| Function | Use |
| --- | --- |
| `build_layered_figure(samples, kind, layers=..., n_clusters=..., cluster_method=..., cluster_mode=..., groups=..., color_values=...)` | PCA, UMAP, Isomap, or similarity with layer controls |
| `prepare_layered_projection_data(samples, kind, layers=...)` | precompute projection coordinates for manual reuse |
| `prepare_kmeans_groups(samples, layers=..., n_clusters=..., cluster_method=...)` | precompute k-means or spectral labels for reuse as categorical colors |
| `build_pair_similarity_figure(samples, layers=...)` | pairwise similarity trajectories |
| `plot_persona_dendrogram(samples, linkage=...)` | hierarchical dendrogram |
| `plot_scree(variance_by_condition, ...)` | PCA explained variance curves |
| `plot_laplacian_eigengap(eigenvalues_by_condition, ...)` | sorted graph-Laplacian eigenvalues for spectral clustering diagnostics |
| `plot_layer_similarity(traces, ...)` | variant cosine by layer |

Example:

```python
from persona_data.synth_persona import SynthPersonaDataset
from persona_vectors.attributes import attribute_color_kwargs
from persona_vectors.plots import build_layered_figure, prepare_layered_projection_data

persona_dataset = SynthPersonaDataset()
projection_data = prepare_layered_projection_data(
    samples,
    "isomap",
    n_components=2,
    graph_overlay=True,
)
fig = build_layered_figure(
    samples,
    "isomap",
    projection_data=projection_data,
    **attribute_color_kwargs(persona_dataset, "age", persona_ids),
)
fig.show()
```

When a UI switches between attributes, reuse `projection_data` and call `build_layered_figure(..., projection_data=projection_data)` with different `attribute_color_kwargs(...)`. Recreate the projection data only when the projection inputs change, such as method, layers, component count, samples, or graph settings.

## Notebooks

`notebooks/unsupervised/manifold.py` and `notebooks/unsupervised/similarity.py` read the published Hub dataset by default and include commented lines for local artifacts.
