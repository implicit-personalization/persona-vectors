# Probes

Linear probes train on saved persona vectors and report how well a single
linear direction or low-rank subspace reads out a persona attribute.

Probe kinds:

- `difference_of_means` — `mean(positive) - mean(negative)` with a midpoint
  bias. Binary only.
- `logistic_regression` — class-balanced, L2-regularized classifier with a
  `StandardScaler`. Binary or multi-class.
- `ridge_regression` — linear regression for ordinal ranks and numeric
  attributes. Ordinal predictions are rounded back to observed ranks.

Evaluation uses one 80/20 train/test split. Classification tasks use a
stratified split; numeric tasks use a plain random split. The scaler and
optional PCA are fit on the train split only. The CLI always excludes
`baseline_assistant` from probe training; saved artifacts are then refit on all
available non-baseline personas.

Core modules:

- `src/persona_vectors/probes.py`
- `src/persona_vectors/plots/probes.py`
- `notebooks/notebook_probes.py`

## CLI

```bash
uv run python main.py probe \
  --model google/gemma-2-9b-it \
  --variant templated \
  --mask-strategy answer_mean \
  --attributes sex born_in_us age \
  --out artifacts/probes
```

The command sweeps each attribute over a representative set of layers, picks
the best `(layer, probe_kind)` by `balanced_accuracy` for classification or
`r2` for numeric attributes, refits that probe on all non-baseline personas,
and writes:

- `probe.json` — schema metadata and held-out evaluation metrics.
- `weights.safetensors` — scaler, optional PCA, linear weights, bias, and the
  diff-of-means direction tensors when applicable.

Use `--all-layers` for an exhaustive layer sweep, `--layers L1 L2 ...` to pin
specific layers, `--pca-components N` to fit a per-split PCA before the probe,
or `--activations-dir artifacts/persona-vectors` to read all-questions vectors.

## Safetensors Artifact

Probe artifacts are saved under:

```text
artifacts/probes/<model_dir>/<mask_strategy>/<variant>/<attribute>/<probe>_layer<layer>/
├── probe.json
└── weights.safetensors
```

For `--pca-components 10`, the final directory is named
`<probe>_pca10_layer<layer>`.

`probe.json` includes `schema_version`, `task`, `probe_kind`,
`n_pca_components`, `normalize_pca`, `layer`, `input_dim`, `artifact_feature_dim`,
`class_names`, and the evaluation metrics. `schema_version == 2` is the
current canonical format. Load it with:

```python
from persona_vectors.probes import load_probe_artifact

artifact = load_probe_artifact("artifacts/probes/.../<probe>_layer20")
metadata = artifact.metadata
tensors = artifact.tensors
```

Consumers should use the metadata and tensors to apply transforms in order:

1. If `scaler_mean` and `scaler_scale` exist, standardize the input vector.
2. If `pca_mean` and `pca_components` exist, center and project with PCA.
3. Apply `weight` and `bias` as the linear head.

Binary classifiers are saved with a two-row `weight` and two-entry `bias` so
the same UI path can handle binary and categorical heads. Diff-of-means
artifacts also include `direction` and `direction_bias` for direct direction
inspection.

## API

```python
from persona_data.synth_persona import SynthPersonaDataset
from persona_vectors.analysis import load_analysis_dataset
from persona_vectors.artifacts import PersonaVectorStore
from persona_vectors.probes import load_probe_artifact, pick_layers, run_attribute_probe

store = PersonaVectorStore(
    "google/gemma-2-9b-it",
    root_dir="artifacts/persona-vectors",
    mask_strategy="answer_mean",
)
dataset = load_analysis_dataset(store, ["templated"])
persona_ids = list(dataset.persona_ids)
samples = dataset.samples("templated")
layers = pick_layers(int(samples.vectors.shape[1]), fast=True)

directory, best_row, task = run_attribute_probe(
    samples,
    SynthPersonaDataset(),
    "sex",
    persona_ids,
    layers=layers,
    model_name="google/gemma-2-9b-it",
    variant="templated",
    mask_strategy="answer_mean",
    output_dir="artifacts/probes",
)

artifact = load_probe_artifact(directory)
```

Lower-level building blocks live in the same module:
`attribute_probe_labels`, `evaluate_classification`, `evaluate_regression`,
`sweep_attribute`, `shuffle_label_baseline`,
`filter_attribute_samples_min_count`, `save_probe_artifact`,
`load_probe_artifact`, `best_row`, and `primary_metric`.

## Plots

`persona_vectors.plots` re-exports the probe plotting helpers.

| Function | Use |
| --- | --- |
| `plot_metric_over_layers(rows, attribute, metric=...)` | Line plot of one held-out metric over layers, one trace per probe kind. |
| `plot_metric_comparison({"full": rows, "pca10": pca_rows}, attribute, metric=...)` | Overlay full-feature and PCA probe sweeps. |
| `plot_attribute_layer_selectivity_heatmap(rows_by_attribute, metric=...)` | Attribute-by-layer heatmap using the best probe kind per cell, optionally subtracting the baseline. |
