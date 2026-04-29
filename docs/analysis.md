# Analysis

Per-layer analyses over saved activations. Analysis views use one mean vector
per persona, computed by averaging that persona's extracted question samples.

Core module: `src/persona_vectors/analysis.py`
Plots: `src/persona_vectors/plots.py`
Reference notebook: `notebooks/notebook_compare.py`

## Functions

All functions operate on tensors loaded from `ActivationStore`. Shape
conventions match the rest of the codebase:

- Saved vectors: `(n_questions, num_layers, hidden_size)`
- Per-persona mean: `(num_layers, hidden_size)`

### `load_persona_mean_samples(..., include_baseline=False)`

Loads saved activation tensors and returns one `LayeredSamples` record per
persona. Each sample keeps the layer axis: `(n_personas, num_layers, hidden_size)`.
Pass `include_baseline=True` to append the persona-less Assistant baseline
sample (loaded from the `baseline` artifact group) as one extra row.

### `load_variant_mean_samples(...)`

Loads one `LayeredSamples` object per prompt variant using the same persona
order for every variant. This is useful for variant-to-variant comparisons.

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

All plot functions return a `go.Figure`. The main entry point is
`build_layered_figure()`, which creates the interactive layer-slider views used
by the CLI. Pass `filename="..."` to the smaller standalone plot functions to
write an HTML artifact under `<artifacts_dir>/plots/<filename>.html`, or
`show=True` to open in the browser.

| Function | Best for |
|---|---|
| `plot_scree(variance_by_condition, n_components=20, cumulative=True)` | Comparing PCA spectra across representative layers |
| `plot_layer_similarity(traces, ...)` | Cosine similarity per layer across prompt variants |
| `build_pair_similarity_figure(samples, layers=None)` | Line trajectories for every persona pair across selected layers |
| `build_layered_figure(samples, kind, layers=None)` | Interactive PCA, UMAP, or similarity figure with layer controls |

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

The plotting code intentionally keeps Plotly-specific layout helpers private to
`plots.py`. Public callers should generally load `LayeredSamples` with
`analysis.py`, then call `build_layered_figure()` or
`build_pair_similarity_figure()`.

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

samples = load_persona_mean_samples(
    "artifacts/activations",
    "google/gemma-2-9b-it",
    "biography",
    "answer_mean",
)

# persona similarity at a middle layer, centered to remove shared DC component
mid = samples.vectors.shape[1] // 2
means = samples.vectors[:, mid, :]
print(cosine_similarity_matrix(means))
print(pca_explained_variance(means))
```
