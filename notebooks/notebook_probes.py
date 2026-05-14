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
FAST = True

torch.set_grad_enabled(False)

store = HFPersonaVectorStore(REPO_ID, MODEL_NAME, mask_strategy=MASK_STRATEGY)
persona_ids = store.list_personas([VARIANT])

samples = load_persona_vectors(store, VARIANT, persona_ids=persona_ids)
persona_dataset = SynthPersonaDataset()
layers = pick_layers(int(samples.vectors.shape[1]), fast=FAST)

# %% Binary — difference_of_means vs logistic_regression
# Swap `attribute` to: born_in_us, mothers_work_history, speak_other_language, us_citizenship_status
attribute = "sex"
labels = attribute_probe_labels(persona_dataset, attribute, persona_ids, task="binary")

# n_pca_components fits a PCA on the train split only (no leakage). Sweep both
# full activations and the 10-component compression, then overlay them in one
# figure to see how much of the attribute lives in the top components.
rows = sweep_attribute(samples, labels, layers=layers)
pca_rows = sweep_attribute(samples, labels, layers=layers, n_pca_components=10)
plot_metric_comparison(
    {"full": rows, "pca10": pca_rows}, attribute, metric="balanced_accuracy"
).show()

best = best_row(rows, "balanced_accuracy")
best_pca = best_row(pca_rows, "balanced_accuracy")
print(
    f"{attribute}: full best bal_acc={best['balanced_accuracy']:.3f} "
    f"(layer={best['layer']}); "
    f"pca10 best bal_acc={best_pca['balanced_accuracy']:.3f} "
    f"(layer={best_pca['layer']})"
)

# %% Categorical — multinomial logistic regression
# Swap `attribute` to: marital_status

attribute = "race"
labels = attribute_probe_labels(
    persona_dataset, attribute, persona_ids, task="categorical"
)
probe_samples, labels = filter_attribute_samples_min_count(
    samples, labels, min_count=MIN_COUNT
)
rows = sweep_attribute(probe_samples, labels, layers=layers)
plot_metric_over_layers(rows, attribute, metric="balanced_accuracy").show()
best = best_row(rows, "balanced_accuracy")
print(
    f"{attribute}: best layer={best['layer']} probe={best['probe_kind']} "
    f"bal_acc={best['balanced_accuracy']:.3f} "
    f"(baseline={best['baseline_balanced_accuracy']:.3f})"
)

# %% Ordinal — ridge on rank, rounded back to integer
# Swap `attribute` to: political_views, total_wealth

attribute = "highest_degree_received"
labels = attribute_probe_labels(persona_dataset, attribute, persona_ids, task="ordinal")
probe_samples, labels = filter_attribute_samples_min_count(
    samples, labels, min_count=MIN_COUNT
)
rows = sweep_attribute(probe_samples, labels, layers=layers)
plot_metric_over_layers(rows, attribute, metric="balanced_accuracy").show()
plot_metric_over_layers(rows, attribute, metric="mae").show()
by_acc = best_row(rows, "balanced_accuracy")
by_mae = best_row(rows, "mae", higher_is_better=False)
print(
    f"{attribute}: by bal_acc layer={by_acc['layer']} "
    f"bal_acc={by_acc['balanced_accuracy']:.3f}; "
    f"by MAE layer={by_mae['layer']} mae={by_mae['mae']:.3f}"
)

# %% Numeric — ridge regression on raw value

attribute = "age"
labels = attribute_probe_labels(persona_dataset, attribute, persona_ids, task="numeric")
rows = sweep_attribute(samples, labels, layers=layers)
plot_metric_over_layers(rows, attribute, metric="r2").show()
plot_metric_over_layers(rows, attribute, metric="mae").show()
by_r2 = best_row(rows, "r2")
by_mae = best_row(rows, "mae", higher_is_better=False)
print(
    f"{attribute}: by R^2 layer={by_r2['layer']} r2={by_r2['r2']:.3f} "
    f"(baseline_r2={by_r2['baseline_r2']:.3f}); "
    f"by MAE layer={by_mae['layer']} mae={by_mae['mae']:.3f} "
    f"(baseline_mae={by_mae['baseline_mae']:.3f})"
)

# %% Attribute × layer selectivity heatmap
# One-glance summary of "where" each attribute is encoded. Per (attribute,
# layer) we take the best probe_kind and subtract the majority-class /
# mean-prediction baseline so different attributes are comparable. Numeric
# attributes need r2 in a second pass.

CLASSIFICATION_ATTRIBUTES = [
    "sex",
    "race",
    "highest_degree_received",
    "political_views",
    "marital_status",
    "us_citizenship_status",
]
classification_rows = {}
for attribute in CLASSIFICATION_ATTRIBUTES:
    task = infer_probe_task(persona_dataset, attribute)
    labels = attribute_probe_labels(persona_dataset, attribute, persona_ids, task=task)
    probe_samples, labels = filter_attribute_samples_min_count(
        samples, labels, min_count=MIN_COUNT
    )
    classification_rows[attribute] = sweep_attribute(
        probe_samples, labels, layers=layers
    )

plot_attribute_layer_selectivity_heatmap(
    classification_rows,
    metric="balanced_accuracy",
).show()
