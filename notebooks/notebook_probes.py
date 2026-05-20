#!/usr/bin/env python

"""Linear probes for persona attributes — one notebook, all task types.

Each section runs a probe sweep over layers for an example attribute:
- binary           (e.g. sex)
- categorical      (e.g. race)
- ordinal          (e.g. highest_degree_received)
- numeric          (e.g. age)

Swap the attribute in any cell to explore others. Evaluation uses one
stratified 80/20 split for classification; scaler/PCA transforms are fit on
the train split only.
"""

# %% Imports

import torch
from persona_data.synth_persona import SynthPersonaDataset
from rich.console import Console
from rich.table import Table

from persona_vectors.analysis import load_persona_vectors
from persona_vectors.artifacts import HFPersonaVectorStore
from persona_vectors.extraction import MaskStrategy
from persona_vectors.plots import (
    plot_attribute_layer_selectivity_heatmap,
    plot_metric_comparison,
    plot_metric_over_layers,
)
from persona_vectors.probes import (
    attribute_probe_labels,
    best_row,
    filter_attribute_samples_min_count,
    infer_probe_task,
    pick_layers,
    sweep_attribute,
)

# %% Setup — shared across all task types

REPO_ID = "implicit-personalization/synth-persona-vectors"
MODEL_NAME = "google/gemma-3-27b-it"
MASK_STRATEGY = MaskStrategy.ANSWER_MEAN
VARIANT = "biography"
MIN_COUNT = 5  # drop classes rarer than this before the 80/20 split
FAST = False  # True: 5 evenly-spaced layers (quick). False: every layer.

torch.set_grad_enabled(False)

store = HFPersonaVectorStore(REPO_ID, MODEL_NAME, mask_strategy=MASK_STRATEGY)
persona_ids = store.list_personas([VARIANT])

samples = load_persona_vectors(store, VARIANT, persona_ids=persona_ids)
persona_dataset = SynthPersonaDataset()
layers = pick_layers(int(samples.vectors.shape[1]), fast=FAST)


def run_probe(attr: str, *, task: str | None = None, **sweep_kwargs):
    """Labels -> min-count filter -> layer sweep for one attribute.

    ``task`` defaults to the dataset's inferred task. The min-count filter is
    a no-op for numeric attributes, so this is uniform across task types.
    """
    task = task or infer_probe_task(persona_dataset, attr)
    labels = attribute_probe_labels(persona_dataset, attr, persona_ids, task=task)
    probe_samples, labels = filter_attribute_samples_min_count(
        samples, labels, min_count=MIN_COUNT
    )
    return sweep_attribute(probe_samples, labels, layers=layers, **sweep_kwargs)


def report_best(rows, label, metric="balanced_accuracy", *, higher_is_better=True):
    """Print the best row for ``metric`` (with baseline when available)."""
    best = best_row(rows, metric, higher_is_better=higher_is_better)
    baseline = best.get(f"baseline_{metric}")
    suffix = f" (baseline={baseline:.3f})" if baseline is not None else ""
    print(
        f"{label}: best layer={best['layer']} probe={best['probe_kind']} "
        f"{metric}={best[metric]:.3f}{suffix}"
    )
    return best


# %% Attribute overview
console = Console()
table = Table(title="Persona Attributes")
table.add_column("Attribute", style="cyan")
table.add_column("Kind", style="magenta")
table.add_column("Unique")
for name in persona_dataset.attribute_names:
    info = persona_dataset.attribute_info(name)
    table.add_row(
        name,
        info.get("kind", ""),
        str(info.get("n_unique_seed_values", "")),
    )
console.print(table)

# %% Binary — difference_of_means vs logistic_regression
# Swap `attribute` to: born_in_us, mothers_work_history, speak_other_language, us_citizenship_status
attributes = ["born_in_us", "mothers_work_history"]

# n_pca_components fits a PCA on the train split only (no leakage). Sweep both
# full activations and the 5-component compression, then overlay them in one
# figure to see how much of the attribute lives in the top components.
# NOTE: difference_of_means only, so the overlay stays readable.
kw = dict(task="binary", probe_kinds=["difference_of_means"])
all_rows, all_pca_rows = [], []
for attr in attributes:
    all_rows.extend(run_probe(attr, **kw))
    all_pca_rows.extend(run_probe(attr, **kw, n_pca_components=5))

plot_metric_comparison(
    {"full": all_rows, "pca5": all_pca_rows}, attributes, metric="balanced_accuracy"
).show()

report_best(all_rows, "full")
report_best(all_pca_rows, "pca5")

# %% Categorical — multinomial logistic regression
# Swap `attribute` to: marital_status

attributes = ["residence_at_16", "detailed_race"]
kw = dict(task="categorical")
for attr in attributes:
    all_rows.extend(run_probe(attr, **kw))
    all_pca_rows.extend(run_probe(attr, **kw, n_pca_components=5))


plot_metric_comparison(
    {"full": all_rows, "pca5": all_pca_rows}, attributes, metric="balanced_accuracy"
).show()
report_best(all_rows, "full")
report_best(all_pca_rows, "pca5")

# %% Ordinal — ridge on rank, rounded back to integer
# Swap `attribute` to: political_views, total_wealth
# bal_acc rewards exact rank; MAE shows how far off — they can pick different layers.

attribute = "highest_degree_received"
rows = run_probe(attribute, task="ordinal")
plot_metric_over_layers(rows, attribute, metric="balanced_accuracy").show()
plot_metric_over_layers(rows, attribute, metric="mae").show()
report_best(rows, attribute, "balanced_accuracy")
report_best(rows, attribute, "mae", higher_is_better=False)

# %% Numeric — ridge regression on raw value
# r2 is scale-free; MAE is in native units (years) and comparable to baseline_mae.

attribute = "age"
rows = run_probe(attribute, task="numeric")
plot_metric_over_layers(rows, attribute, metric="r2").show()
plot_metric_over_layers(rows, attribute, metric="mae").show()
report_best(rows, attribute, "r2")
report_best(rows, attribute, "mae", higher_is_better=False)

# %% Attribute × layer selectivity heatmap
# One-glance summary of "where" each attribute is encoded.
# Per (attribute, layer) we take the best probe_kind and subtract the
# majority-class / mean-prediction baseline so different attributes are comparable.
# Numeric attributes need r2 in a second pass.

CLASSIFICATION_ATTRIBUTES = [
    "sex",
    "race",
    "highest_degree_received",
    "political_views",
    "marital_status",
    "us_citizenship_status",
]
classification_rows = {attr: run_probe(attr) for attr in CLASSIFICATION_ATTRIBUTES}

plot_attribute_layer_selectivity_heatmap(
    classification_rows,
    metric="balanced_accuracy",
).show()
