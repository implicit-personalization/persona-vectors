# Probes

Linear probes train on saved persona vectors and report how well a single direction (or low-rank subspace) reads out a persona attribute. Three probe kinds, all linear:

- `difference_of_means` — `mean(positive) - mean(negative)` with a midpoint bias. Anthropic-style persona-vector direction. Binary only.
- `logistic_regression` — class-balanced, L2-regularized, with a `StandardScaler`. Binary or multi-class.
- `ridge_regression` — for ordinal ranks and numeric attributes. Ordinal predictions are rounded back to the rank scale.

Cross-validation is 5-fold by default (`StratifiedKFold` for classification, `KFold` for regression). The scaler and optional PCA fit *inside each fold*, so no leakage from held-out personas. Final saved artifacts are refit on all personas.

Core modules:

- `src/persona_vectors/probes.py`
- `src/persona_vectors/plots/probes.py` (figures)
- `notebooks/probes/` (per-task notebooks: binary, categorical, ordinal, numeric)

## CLI

```bash
uv run python main.py probe \
  --model google/gemma-2-9b-it \
  --variant templated \
  --mask-strategy answer_mean \
  --attributes sex born_in_us age \
  --out artifacts/probes
```

The command sweeps each attribute over a representative set of layers, picks the
best (layer, probe_kind, feature_space) by `balanced_accuracy` (classification)
or `r2` (numeric), refits on all personas, and writes:

- `probe.json` — schema metadata + CV metrics
- `weights.safetensors` — scaler, optional PCA, linear weights, plus the diff-of-means direction when applicable
- `probe.pt` — persona-ui-compatible payload (binary & categorical, raw feature space only)

Pass `--all-layers` for an exhaustive sweep, `--layers L1 L2 ...` to pin specific layers, `--feature-spaces raw pca10` to compare a low-rank baseline, or `--activations-dir artifacts/persona-vectors` to read the all-questions tree.

## API

```python
from persona_vectors.analysis import load_persona_vectors
from persona_vectors.artifacts import PersonaVectorStore
from persona_vectors.probes import pick_layers, run_attribute_probe
from persona_data.synth_persona import SynthPersonaDataset

store = PersonaVectorStore(
    "google/gemma-2-9b-it",
    root_dir="artifacts/persona-vectors",
    mask_strategy="answer_mean",
)
persona_ids = store.list_personas(["templated"])
samples = load_persona_vectors(store, "templated", persona_ids=persona_ids)
layers = pick_layers(int(samples.vectors.shape[1]), fast=True)

artifact, best_row, task = run_attribute_probe(
    samples,
    SynthPersonaDataset(),
    "sex",
    persona_ids,
    layers=layers,
    feature_spaces=["raw"],
    model_name="google/gemma-2-9b-it",
    variant="templated",
    mask_strategy="answer_mean",
    output_dir="artifacts/probes",
)
```

Lower-level building blocks (used by the notebooks) live in the same module:
`attribute_probe_labels`, `sweep_attribute`, `cross_validate_classification`,
`cross_validate_regression`, `shuffle_label_baseline`,
`filter_attribute_samples_min_count`, `save_probe_artifact`, `best_row`, and
`primary_metric`.

## Notebooks

`notebooks/probes/{binary,categorical,ordinal,numeric}.py` each load persona
vectors from the Hub by default. To iterate against locally extracted vectors
(the same tree that `extraction_all_questions.sh` produces and that steering /
persona-ui consume), swap `HFPersonaVectorStore` for `PersonaVectorStore` with
`root_dir="artifacts/persona-vectors"`.

## Plots

`persona_vectors.plots` re-exports one probe-specific helper, also available
as `persona_vectors.plots.probes`:

| Function | Use |
| --- | --- |
| `plot_metric_over_layers(rows, attribute, metric=...)` | Line plot of one CV metric per layer, one trace per (probe_kind, feature_space). Draws the baseline as a dotted horizontal line. |
