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
    project_pca,
    project_umap,
)

layer_vectors = samples.vectors[:, 20, :]
similarity = cosine_similarity_matrix(layer_vectors)
pca = project_pca(layer_vectors, n_components=2)
variance = pca_explained_variance(layer_vectors)
```

Cosine similarity is centered by default to remove the shared residual-stream component that otherwise pushes raw cosine values toward 1.

## Clustering

```python
from persona_vectors.analysis import (
    cluster_hdbscan,
    prepare_layer_mean_cluster_samples,
)

cluster_input = prepare_layer_mean_cluster_samples(samples.vectors)
labels = cluster_hdbscan(cluster_input, center=False, normalize=False)
```

Available helpers:

| Function | Method |
| --- | --- |
| `cluster_kmeans(samples, n_clusters)` | k-means |
| `cluster_agglomerative(samples, n_clusters, linkage=...)` | hierarchical clustering |
| `cluster_agglomerative_ward(samples, n_clusters)` | Ward wrapper |
| `cluster_hdbscan(samples, min_cluster_size=...)` | HDBSCAN, with `-1` for noise |

By default, clustering inputs are centered and L2-normalized. For PCA/UMAP colors, `build_layered_figure(..., n_clusters=k)` runs k-means directly. Set `cluster_method="agglomerative"` for hierarchical agglomerative clustering, or `cluster_method="hdbscan"` with `min_cluster_size=...` for density clustering with `"Noise"` labels. Use `cluster_mode="mean_across_layers"` for stable colors from the centered/unit mean vector per persona, `cluster_mode="first_layer"` for stable colors from the first plotted layer, or `cluster_mode="per_layer"` to recompute clustering for every layer frame. You can still pass explicit labels as `groups=...`; for per-layer custom labels, pass a `{layer: labels}` mapping.

## Plot Helpers

All plot helpers return a Plotly `go.Figure`.

| Function | Use |
| --- | --- |
| `build_layered_figure(samples, kind, layers=..., n_clusters=..., cluster_mode=..., groups=...)` | PCA, UMAP, or similarity with layer controls |
| `build_pair_similarity_figure(samples, layers=...)` | pairwise similarity trajectories |
| `plot_persona_dendrogram(samples, linkage=...)` | hierarchical dendrogram |
| `plot_scree(variance_by_condition, ...)` | PCA explained variance curves |
| `plot_layer_similarity(traces, ...)` | variant cosine by layer |
| `plot_hdbscan_cluster_counts(samples_by_variant, ...)` | per-layer cluster counts |

Example:

```python
from persona_vectors.plots import build_layered_figure

fig = build_layered_figure(samples, "pca", n_components=3)
fig.show()
```

## Notebooks

`notebooks/notebook_pca.py` and `notebooks/notebook_similarity.py` read the published Hub dataset by default and include commented lines for local artifacts.
