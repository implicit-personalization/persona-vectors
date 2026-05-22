#!/usr/bin/env python
"""Sweep every persona attribute over all layers and print a results table.

Best row per attribute (no fast layer subsampling). Binary reports the best
score for each probe kind; categorical/ordinal are ranked by balanced_accuracy;
numeric by r2. Grouped by inferred task type.
"""

import torch
from persona_data.synth_persona import SynthPersonaDataset
from tqdm import tqdm

from persona_vectors.analysis import load_persona_vectors
from persona_vectors.artifacts import HFPersonaVectorStore
from persona_vectors.extraction import MaskStrategy
from persona_vectors.probes import (
    attribute_probe_labels,
    best_row,
    filter_attribute_samples_min_count,
    infer_probe_task,
    pick_layers,
    primary_metric,
    sweep_attribute,
)

REPO_ID = "implicit-personalization/synth-persona-vectors"
MODEL_NAME = "google/gemma-3-27b-it"
MASK_STRATEGY = MaskStrategy.ANSWER_MEAN
VARIANT = "biography"
MIN_COUNT = 5

torch.set_grad_enabled(False)

store = HFPersonaVectorStore(REPO_ID, MODEL_NAME, mask_strategy=MASK_STRATEGY)
persona_ids = store.list_personas([VARIANT])
samples = load_persona_vectors(store, VARIANT, persona_ids=persona_ids)
persona_dataset = SynthPersonaDataset()
layers = pick_layers(int(samples.vectors.shape[1]), fast=False)

TASK_ORDER = ["binary", "categorical", "ordinal", "numeric"]
results: dict[str, list[dict]] = {t: [] for t in TASK_ORDER}

for attr in tqdm(persona_dataset.attribute_names, desc="attributes", unit="attr"):
    task = infer_probe_task(persona_dataset, attr)
    labels = attribute_probe_labels(persona_dataset, attr, persona_ids, task=task)
    probe_samples, labels = filter_attribute_samples_min_count(
        samples, labels, min_count=MIN_COUNT
    )
    probe_kinds = (
        ["difference_of_means", "logistic_regression"] if task == "binary" else None
    )
    rows = sweep_attribute(
        probe_samples, labels, layers=layers, probe_kinds=probe_kinds
    )
    metric = primary_metric(task)
    n_classes = len(labels.class_names) if labels.class_names else None
    if task == "binary":
        for probe_kind in ["difference_of_means", "logistic_regression"]:
            best = best_row(
                [row for row in rows if row["probe_kind"] == probe_kind], metric
            )
            results[task].append(
                {
                    "attribute": attr,
                    "n": len(labels.y),
                    "classes": n_classes,
                    "layer": int(best["layer"]),
                    "probe": best["probe_kind"],
                    "metric": metric,
                    "value": float(best[metric]),
                    "baseline": best.get(f"baseline_{metric}"),
                    "mae": best.get("mae"),
                }
            )
    else:
        best = best_row(rows, metric)
        results[task].append(
            {
                "attribute": attr,
                "n": len(labels.y),
                "classes": n_classes,
                "layer": int(best["layer"]),
                "probe": best["probe_kind"],
                "metric": metric,
                "value": float(best[metric]),
                "baseline": best.get(f"baseline_{metric}"),
                "mae": best.get("mae"),
            }
        )

for task in TASK_ORDER:
    block = results[task]
    if not block:
        continue
    print(f"\n=== {task.upper()} ===")
    if task == "numeric":
        print(f"{'attribute':<26}{'n':>5}{'layer':>7}{'r2':>8}{'mae':>9}{'base_r2':>9}")
        for r in block:
            print(
                f"{r['attribute']:<26}{r['n']:>5}{r['layer']:>7}"
                f"{r['value']:>8.3f}{(r['mae'] or 0):>9.2f}{(r['baseline'] or 0):>9.3f}"
            )
    else:
        print(
            f"{'attribute':<26}{'n':>5}{'cls':>5}{'layer':>7}"
            f"{'probe':>22}{'bal_acc':>9}{'baseline':>10}"
        )
        for r in block:
            print(
                f"{r['attribute']:<26}{r['n']:>5}{r['classes']:>5}{r['layer']:>7}"
                f"{r['probe']:>22}{r['value']:>9.3f}{(r['baseline'] or 0):>10.3f}"
            )
