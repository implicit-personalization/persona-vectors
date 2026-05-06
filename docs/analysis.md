# Analysis

Per-layer analyses over saved activations. Analysis views use one mean vector
per persona, computed by averaging that persona's extracted question samples.

Core module: `src/persona_vectors/analysis.py`
Plots: `src/persona_vectors/plots.py`
Reference notebook: `notebooks/notebook_compare.py`

## Analysis Functions

These lower-level functions operate on tensors loaded from `ActivationStore`.
Shape conventions match the rest of the codebase:

- Saved vectors: `(n_questions, num_layers, hidden_size)`
- Per-persona mean: `(num_layers, hidden_size)`

### `load_persona_mean_samples(...)`

Loads saved activation tensors and returns the mean vectors used by plotting
and numerical analysis. Most UI/notebook code should call the plot helpers
below instead of handling this object directly.

### `load_variant_mean_samples(...)`

Loads mean vectors per prompt variant using the same persona order for every
variant. This is useful for custom variant-to-variant comparisons.

```python
from persona_vectors.analysis import load_variant_mean_samples
from persona_vectors.artifacts import ActivationStore

store = ActivationStore("google/gemma-2-9b-it", mask_strategy="answer_mean")
variants = store.available_variants()
persona_ids = store.list_personas(variants)

samples_by_variant = load_variant_mean_samples(
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

## Plots

All plot functions return a `go.Figure`. Load saved activations once with the
analysis helpers, then pass the resulting `LayeredSamples` to the plot builders.

| Function | Best for |
|---|---|
| `plot_scree(variance_by_condition, n_components=20, cumulative=True)` | Comparing PCA spectra across representative layers |
| `plot_layer_similarity(traces, ...)` | Cosine similarity per layer across prompt variants |
| `build_layered_figure(samples, kind, layers=...)` | Interactive PCA, UMAP, or similarity figure with layer controls |
| `build_pair_similarity_figure(samples, layers=...)` | Line trajectories for every persona pair across selected layers |

## CLI

```bash
python main.py analyze \
  --model google/gemma-2-9b-it \
  --variant biography \
  --mask-strategy answer_mean \
  --out artifacts/plots
```

This writes interactive HTML files with layer dropdowns:

- `persona_mean_pca`: one point per persona, averaged over questions
- `persona_mean_similarity`: centered persona cosine heatmap by layer
- `persona_pair_similarity`: persona-pair similarity trajectories across layers
- `pca_scree`: PCA explained-variance curves for selected or representative layers

For prompt-only extraction, use `persona_mean` or `persona_last` during
extraction and analysis. Those strategies run on the rendered system prompt
only, so they do not need every QA pair.

## Reference notebook

### `notebook_compare.py`

Layer-wise variant similarity across saved prompt variants.

## Quick example

```python
from persona_vectors.analysis import (
    cosine_similarity_matrix,
    load_persona_mean_samples,
    pca_explained_variance,
)
from persona_vectors.artifacts import ActivationStore

store = ActivationStore("google/gemma-2-9b-it", mask_strategy="answer_mean")
samples = load_persona_mean_samples(
    store,
    "biography",
)

# persona similarity at a middle layer, centered to remove shared DC component
mid = samples.vectors.shape[1] // 2
means = samples.vectors[:, mid, :]
print(cosine_similarity_matrix(means))
print(pca_explained_variance(means))
```

```python
from persona_vectors.analysis import load_persona_mean_samples
from persona_vectors.artifacts import ActivationStore
from persona_vectors.plots import build_layered_figure

store = ActivationStore("google/gemma-2-9b-it", mask_strategy="answer_mean")
persona_ids = store.list_personas(["biography"])
samples = load_persona_mean_samples(store, "biography", persona_ids=persona_ids)

fig = build_layered_figure(
    samples,
    "similarity",
)
fig.show()
```
