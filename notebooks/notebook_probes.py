#!/usr/bin/env python

"""Linear probes for persona attributes — one notebook, all task types.

Each section runs a probe sweep over layers for an example attribute:
- binary           (e.g. sex)
- categorical      (e.g. race)
- ordinal          (e.g. highest_degree_received)
- numeric          (e.g. age)

Swap the attribute in any cell to explore others. 5-fold CV; scaler refit
per fold.
"""

# %% Imports

import torch
from persona_data.synth_persona import BASELINE_PERSONA_ID, SynthPersonaDataset

from persona_vectors.analysis import list_comparison_personas, load_persona_vectors
from persona_vectors.artifacts import HFPersonaVectorStore
from persona_vectors.extraction import MaskStrategy
from persona_vectors.plots import plot_metric_over_layers
from persona_vectors.probes import (
    attribute_probe_labels,
    best_row,
    filter_attribute_samples_min_count,
    pick_layers,
    sweep_attribute,
)

# %% Setup — shared across all task types

REPO_ID = "implicit-personalization/synth-persona-vectors"
MODEL_NAME = "google/gemma-3-27b-it"
MASK_STRATEGY = MaskStrategy.ANSWER_MEAN
VARIANT = "templated"
N_SPLITS = 5
FAST = True

torch.set_grad_enabled(False)

store = HFPersonaVectorStore(REPO_ID, MODEL_NAME, mask_strategy=MASK_STRATEGY)
persona_ids = [
    pid
    for pid in list_comparison_personas(store, [VARIANT])
    if pid != BASELINE_PERSONA_ID
]
samples = load_persona_vectors(store, VARIANT, persona_ids=persona_ids)
persona_dataset = SynthPersonaDataset()
layers = pick_layers(int(samples.vectors.shape[1]), fast=FAST)

# %% Binary — difference_of_means vs logistic_regression
# Swap `attribute` to: born_in_us, mothers_work_history, speak_other_language,
# us_citizenship_status

attribute = "sex"
labels = attribute_probe_labels(persona_dataset, attribute, persona_ids, task="binary")
sweep = sweep_attribute(
    samples,
    labels,
    layers=layers,
    probe_kinds=["difference_of_means", "logistic_regression"],
)
plot_metric_over_layers(sweep.rows, attribute, metric="balanced_accuracy").show()

best = best_row(sweep.rows, "balanced_accuracy")
print(
    f"{attribute}: best {best['probe_kind']}/{best['feature_space']} "
    f"layer={best['layer']} bal_acc={best['balanced_accuracy']:.3f}"
)

# %% Categorical — multinomial logistic regression
# Swap `attribute` to: marital_status

attribute = "race"
labels = attribute_probe_labels(
    persona_dataset, attribute, persona_ids, task="categorical"
)
probe_samples, labels = filter_attribute_samples_min_count(
    samples, labels, min_count=N_SPLITS
)
sweep = sweep_attribute(probe_samples, labels, layers=layers, n_splits=N_SPLITS)
plot_metric_over_layers(sweep.rows, attribute, metric="balanced_accuracy").show()
best = best_row(sweep.rows, "balanced_accuracy")
print(
    f"{attribute}: best layer={best['layer']} space={best['feature_space']} "
    f"bal_acc={best['balanced_accuracy']:.3f} "
    f"(baseline={best['baseline_balanced_accuracy']:.3f})"
)

# %% Ordinal — ridge on rank, rounded back to integer
# Swap `attribute` to: political_views, total_wealth

attribute = "highest_degree_received"
labels = attribute_probe_labels(persona_dataset, attribute, persona_ids, task="ordinal")
probe_samples, labels = filter_attribute_samples_min_count(
    samples, labels, min_count=N_SPLITS
)
sweep = sweep_attribute(probe_samples, labels, layers=layers, n_splits=N_SPLITS)
plot_metric_over_layers(sweep.rows, attribute, metric="balanced_accuracy").show()
plot_metric_over_layers(sweep.rows, attribute, metric="mae").show()
by_acc = best_row(sweep.rows, "balanced_accuracy")
by_mae = best_row(sweep.rows, "mae", higher_is_better=False)
print(
    f"{attribute}: by bal_acc layer={by_acc['layer']} "
    f"bal_acc={by_acc['balanced_accuracy']:.3f}; "
    f"by MAE layer={by_mae['layer']} mae={by_mae['mae']:.3f}"
)

# %% Numeric — ridge regression on raw value

attribute = "age"
labels = attribute_probe_labels(persona_dataset, attribute, persona_ids, task="numeric")
sweep = sweep_attribute(samples, labels, layers=layers)
plot_metric_over_layers(sweep.rows, attribute, metric="r2").show()
plot_metric_over_layers(sweep.rows, attribute, metric="mae").show()
by_r2 = best_row(sweep.rows, "r2")
by_mae = best_row(sweep.rows, "mae", higher_is_better=False)
print(
    f"{attribute}: by R^2 layer={by_r2['layer']} r2={by_r2['r2']:.3f} "
    f"(baseline_r2={by_r2['baseline_r2']:.3f}); "
    f"by MAE layer={by_mae['layer']} mae={by_mae['mae']:.3f} "
    f"(baseline_mae={by_mae['baseline_mae']:.3f})"
)
