# Analysis

Per-layer analyses over saved activations. Analysis views use one saved vector
per persona; extraction has already averaged across that persona's QA samples.

Core module: `src/persona_vectors/analysis.py`
Plots: `src/persona_vectors/plots.py`
Reference notebooks: `notebooks/notebook_pca.py`, `notebooks/notebook_similarity.py`

## Analysis Functions

These lower-level functions operate on tensors loaded from `ActivationStore` or
`HFActivationStore`.
Shape conventions match the rest of the codebase:

- Saved persona vector: `(num_layers, hidden_size)`
- Layered sample collection: `(n_personas, num_layers, hidden_size)`

### `load_persona_vectors(...)`

Loads saved activation tensors and returns the persona vectors used by plotting
and numerical analysis. Most UI/notebook code should call the plot helpers
below instead of handling this object directly.

### `load_variant_vectors(...)`

Loads persona vectors per prompt variant using the same persona order for every
variant. This is useful for custom variant-to-variant comparisons.

```python
from persona_vectors.analysis import load_variant_vectors
from persona_vectors.artifacts import ActivationStore

store = ActivationStore("google/gemma-2-9b-it", mask_strategy="answer_mean")
variants = store.available_variants()
persona_ids = store.list_personas(variants)

samples_by_variant = load_variant_vectors(
    store,
    variants,
    persona_ids=persona_ids,
)
```

### `cosine_similarity_matrix(samples, center=True)`

Cosine similarity for a 2-D `(n_samples, hidden_size)` matrix; returns
`(n, n)` tensor. Centered by default — the per-feature mean across rows is
subtracted before normalising. LLM residual-stream means share a large DC
component that pushes every raw pairwise cosine toward ~1; centering removes
it so the actual persona cluster structure shows.

### `project_pca(samples)` / `project_umap(samples)`

2-D projection for scatter visualisation. Input: `(n_samples, hidden_size)`.
Returns `(n_samples, 2)`. `project_pca` uses sklearn's `PCA` (centers
features by default); `project_umap` centers explicitly before fitting.

### `pca_explained_variance(samples, n_components=None)`

Explained-variance ratio per principal component. Input `(n_samples,
hidden_size)`. Returns a 1-D numpy array of length `n_components` (default:
`min(n_samples, hidden_size)`).

### Clustering

Three small wrappers around scikit-learn estimators for grouping personas in
activation space. All take a 2-D `(n_samples, hidden_size)` tensor and return
a 1-D numpy array of integer labels.

| Function | Method | Notes |
|---|---|---|
| `cluster_kmeans(samples, n_clusters, *, seed=0)` | K-means (k-means++ init) | Requires `k`. Deterministic for fixed `seed`. |
| `cluster_agglomerative_ward(samples, n_clusters)` | Hierarchical, Ward linkage | Requires `k`. Merges by minimising within-cluster variance. |
| `cluster_hdbscan(samples, *, min_cluster_size=2, min_samples=None)` | HDBSCAN | No `k`. Returns `-1` for noise points. |

These pair with `build_layered_figure(..., groups=...)` to color a PCA/UMAP
view by any clustering. Cluster once on the per-persona mean across layers
(`samples.vectors.mean(dim=1)`) so each persona keeps a stable color across
every frame; per-layer re-clustering would shuffle colors arbitrarily as you
slide.

```python
from persona_vectors.analysis import cluster_hdbscan, load_persona_vectors
from persona_vectors.artifacts import ActivationStore
from persona_vectors.plots import build_layered_figure

store = ActivationStore("google/gemma-2-9b-it", mask_strategy="answer_mean")
samples = load_persona_vectors(store, "biography")
labels = cluster_hdbscan(samples.vectors.mean(dim=1), min_cluster_size=2)
groups = ["Noise" if c == -1 else f"Cluster {c}" for c in labels]

build_layered_figure(samples, "pca", n_components=3, groups=groups).show()
```

## Plots

All plot functions return a `go.Figure`. Load saved activations once with the
analysis helpers, then pass the resulting `LayeredSamples` to the plot builders.

| Function | Best for |
|---|---|
| `plot_scree(variance_by_condition, n_components=20, cumulative=True)` | Comparing PCA spectra across representative layers |
| `plot_layer_similarity(traces, ...)` | Cosine similarity per layer across prompt variants |
| `build_layered_figure(samples, kind, layers=..., n_clusters=..., groups=...)` | Interactive PCA, UMAP, or similarity figure with layer controls. Pass `n_clusters=k` for a quick k-means coloring, or `groups=[...]` to color by labels from any clustering helper. |
| `build_pair_similarity_figure(samples, layers=...)` | Line trajectories for every persona pair across selected layers |
| `plot_persona_dendrogram(samples, layer=..., layered=True, layers=...)` | Ward dendrogram over personas. Defaults to the per-persona mean across layers; pass `layer=N` for a single-layer view, or `layered=True` for an interactive per-layer slider. Width auto-scales with persona count. |
| `plot_hdbscan_cluster_counts(samples_by_variant, min_cluster_size=...)` | Per-layer HDBSCAN cluster count, one trace per variant. Useful for checking whether persona structure becomes more or less clustered across depth. Hover shows the noise-point count. |

## CLI

```bash
python main.py analyze \
  --model google/gemma-2-9b-it \
  --variant biography \
  --mask-strategy answer_mean \
  --out artifacts/plots
```

This writes interactive HTML files with layer dropdowns:

- `persona_vector_pca`: one point per saved persona vector
- `persona_vector_similarity`: centered persona cosine heatmap by layer
- `persona_pair_similarity`: persona-pair similarity trajectories across layers
- `pca_scree`: PCA explained-variance curves for selected or representative layers

For prompt-only extraction, use `persona_mean` or `persona_last` during
extraction and analysis. Those strategies run on the rendered system prompt
only, so they do not need every QA pair.

## Reference notebooks

Both read the published Hub dataset by default and include commented lines for
switching to local artifacts.

### `notebook_pca.py`

PCA scree, then 3D PCA colored by k-means, Ward, and HDBSCAN clusters. Ends
with the per-layer HDBSCAN cluster-count summary (commented out by default,
since it re-runs HDBSCAN on every layer).

### `notebook_similarity.py`

Centered persona-similarity heatmaps, hierarchical dendrograms with layer
sliders, per-persona pair similarity trajectories, and prompt-variant cosine
comparisons.

## Quick example

```python
from persona_vectors.analysis import (
    cosine_similarity_matrix,
    load_persona_vectors,
    pca_explained_variance,
)
from persona_vectors.artifacts import ActivationStore

store = ActivationStore("google/gemma-2-9b-it", mask_strategy="answer_mean")
samples = load_persona_vectors(
    store,
    "biography",
)

# persona similarity at a middle layer, centered to remove shared DC component
mid = samples.vectors.shape[1] // 2
layer_vectors = samples.vectors[:, mid, :]
print(cosine_similarity_matrix(layer_vectors))
print(pca_explained_variance(layer_vectors))
```

```python
from persona_vectors.analysis import load_persona_vectors
from persona_vectors.artifacts import ActivationStore
from persona_vectors.plots import build_layered_figure

store = ActivationStore("google/gemma-2-9b-it", mask_strategy="answer_mean")
persona_ids = store.list_personas(["biography"])
samples = load_persona_vectors(store, "biography", persona_ids=persona_ids)

fig = build_layered_figure(
    samples,
    "similarity",
)
fig.show()
```
