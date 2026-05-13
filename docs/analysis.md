# Analysis

Analysis operates on saved persona vectors. A saved vector has shape `(num_layers, hidden_size)`; a loaded sample collection has shape `(n_personas, num_layers, hidden_size)`.

Core modules:

- `src/persona_vectors/analysis.py`
- `src/persona_vectors/plots.py`

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
from persona_vectors.artifacts import ActivationStore

store = ActivationStore("google/gemma-2-9b-it", mask_strategy="answer_mean")

samples = load_persona_vectors(store, "biography")
by_variant = load_variant_vectors(store, ["biography", "templated"])
```

`load_variant_vectors()` keeps the persona order shared across variants.

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

Cosine similarity is centered by default to remove the shared residual-stream component that otherwise pushes raw cosine values toward 1.

## Clustering

```python
from persona_vectors.analysis import (
    cluster_agglomerative,
    cluster_kmeans,
    prepare_layer_mean_cluster_samples,
)

cluster_input = prepare_layer_mean_cluster_samples(samples.vectors)
labels = cluster_kmeans(cluster_input, n_clusters=5, center=False, normalize=False)
dendrogram_like_labels = cluster_agglomerative(
    cluster_input,
    n_clusters=5,
    center=False,
    normalize=False,
)
```

Available helpers:

| Function | Method |
| --- | --- |
| `cluster_kmeans(samples, n_clusters)` | k-means |
| `cluster_agglomerative(samples, n_clusters, linkage=...)` | hierarchical clustering |

By default, clustering inputs are centered and L2-normalized. For PCA/UMAP/Isomap cluster colors, `build_layered_figure(..., n_clusters=k)` runs k-means directly. Use `cluster_mode="mean_across_layers"` for stable colors from the centered/unit mean vector per persona, `cluster_mode="first_layer"` for stable colors from the first plotted layer, or `cluster_mode="per_layer"` to recompute clustering for every layer frame. For UI code, call `prepare_kmeans_groups(...)` once and pass the result as `groups=...` when recoloring an existing projection. Dendrograms use hierarchical linkage via `plot_persona_dendrogram(..., linkage=...)`. You can still pass explicit categorical labels as `groups=...`; for numeric or ordinal colors, pass `color_values=...`.

## Plot Helpers

All plot helpers return a Plotly `go.Figure`.

| Function | Use |
| --- | --- |
| `build_layered_figure(samples, kind, layers=..., n_clusters=..., cluster_mode=..., groups=..., color_values=...)` | PCA, UMAP, Isomap, or similarity with layer controls |
| `prepare_layered_projection_data(samples, kind, layers=...)` | precompute projection coordinates for manual reuse |
| `prepare_kmeans_groups(samples, layers=..., n_clusters=...)` | precompute k-means labels for reuse as categorical colors |
| `build_pair_similarity_figure(samples, layers=...)` | pairwise similarity trajectories |
| `plot_persona_dendrogram(samples, linkage=...)` | hierarchical dendrogram |
| `plot_scree(variance_by_condition, ...)` | PCA explained variance curves |
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

`notebooks/notebook_manifold.py` and `notebooks/notebook_similarity.py` read the published Hub dataset by default and include commented lines for local artifacts.
