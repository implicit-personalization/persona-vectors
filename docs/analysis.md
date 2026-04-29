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

### `load_persona_mean_samples(...)`

Loads saved activation tensors and returns one `LayeredSamples` record per
persona. Each sample keeps the layer axis: `(n_personas, num_layers, hidden_size)`.

### `pairwise_cosine_similarity(vectors, center=False)`

Cosine similarity matrix between a list of 1-D vectors (one vector per
persona or condition). Returns `(n, n)` tensor.

Pass `center=True` to subtract the grand-mean vector across the list before
normalising. LLM residual-stream means share a large DC component that
pushes every raw pairwise cosine toward ~1; centering removes it so the
actual persona cluster structure shows.

### `project_pca(samples)` / `project_umap(samples)`

2-D projection for scatter visualisation. Input: `(n_samples, hidden_size)`.
Returns `(n_samples, 2)`. `project_pca` uses sklearn's `PCA`, which centers
features by default.

### `pca_explained_variance(samples, n_components=None)`

Explained-variance ratio per principal component. Input `(n_samples,
hidden_size)`. Returns a 1-D numpy array of length `n_components` (default:
`min(n_samples, hidden_size)`).

Use case: detect whether persona conditioning compresses the activation space
(fewer PCs capture most variance). Drives `plot_scree()`.

## Plots

All plot functions return a `go.Figure`. Pass `filename="..."` to write an
HTML artifact under `<artifacts_dir>/plots/<filename>.html`, or `show=True`
to open in the browser.

| Function | Best for |
|---|---|
| `plot_scree(variance_by_condition, n_components=20, cumulative=True)` | Comparing PC spectra across conditions at a single layer |
| `plot_layer_similarity(traces, ...)` | Cosine similarity per layer across prompt variants |
| `plot_similarity_matrix(sim_matrix, labels)` | Single pairwise-cosine heatmap (feed a centered matrix for readability) |
| `plot_similarity_matrix_grid(matrices, labels, titles)` | 2×2 grid across four layers |
| `build_layered_figure(samples, kind, layers=None)` | Interactive PCA, UMAP, or similarity figure with a layer dropdown |
| `build_embedding_figure(coords, labels, ...)` | Scatter plot from projected coordinates |

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

For prompt-only extraction, use `persona_mean` or `persona_last` during
extraction and analysis. Those strategies run on the rendered system prompt
only, so they do not need every QA pair.

## Reference notebook

### `notebook_compare.py`

Layer-wise variant similarity plus a PCA scree view and centered pairwise
persona heatmaps at a chosen layer.

## Quick example

```python
import torch
from persona_vectors.analysis import pairwise_cosine_similarity, pca_explained_variance
from persona_vectors.artifacts import ActivationStore, list_personas

store = ActivationStore("google/gemma-2-9b-it")
persona_ids = list_personas(store.root_dir, "google/gemma-2-9b-it", ["biography"])

# saved question samples for each persona
acts = {pid: store.load("biography", pid)[0] for pid in persona_ids}

# PCA scree at a middle layer, pooled across personas
mid = next(iter(acts.values())).shape[1] // 2
pooled = torch.cat([a[:, mid, :] for a in acts.values()], dim=0)
print(pca_explained_variance(pooled))

# persona similarity, centered to remove shared DC component
means = [a[:, mid, :].mean(dim=0) for a in acts.values()]
print(pairwise_cosine_similarity(means, center=True))
```
