"""Linear probes over persona vectors.

Three probe kinds:

- ``difference_of_means`` — Anthropic-style persona-vector direction with a
  midpoint bias. Closed-form; binary only.
- ``logistic_regression`` — class-balanced, L2-regularized, with a
  StandardScaler. Binary and multi-class.
- ``ridge_regression`` — linear regression for ordinal ranks (predictions
  rounded back to ranks) and numeric attributes.

A single 80/20 train/test split (``random_state=0`` by default), stratified
for classification. Scaler and optional PCA are fit on the train split only to
prevent leakage. Final saved artifacts are refit on all personas.

PCA is an optional dimensionality knob (``n_pca_components``), not a sweep
axis: pass an int to compress features, leave it ``None`` for raw activations.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from safetensors.torch import save_file
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from persona_vectors.analysis import LayeredSamples
from persona_vectors.artifacts import model_dir_name, normalize_mask_strategy

ProbeTask = Literal["binary", "ordinal", "categorical", "numeric"]
ProbeKind = Literal["difference_of_means", "logistic_regression", "ridge_regression"]

CLASSIFICATION_TASKS: frozenset[ProbeTask] = frozenset(
    {"binary", "categorical", "ordinal"}
)

TEST_SIZE = 0.2
RANDOM_STATE = 0


@dataclass(frozen=True)
class AttributeLabels:
    """Stable label container for one persona attribute."""

    attribute_name: str
    task: ProbeTask
    y: np.ndarray
    labels: list[str]
    class_names: list[str] | None = None


# ---------------------------------------------------------------------------
# Task inference and labels
# ---------------------------------------------------------------------------


def infer_probe_task(persona_dataset, attribute_name: str) -> ProbeTask:
    """Infer the simplest probe task from the persona schema."""
    info = persona_dataset.attribute_info(attribute_name)
    kind = info.get("kind")
    if kind == "numeric":
        return "numeric"
    if kind == "ordinal":
        return "ordinal"
    values = info.get("ordered_values")
    if values is not None and len(values) == 2:
        return "binary"
    if kind == "binary":
        return "binary"
    return "categorical"


def attribute_probe_labels(
    persona_dataset,
    attribute_name: str,
    persona_ids: list[str],
    task: ProbeTask | None = None,
) -> AttributeLabels:
    """Return stable labels for the appropriate probe task."""
    task = infer_probe_task(persona_dataset, attribute_name) if task is None else task
    raw_values = list(persona_dataset.attribute_values(attribute_name, persona_ids))
    labels = [str(value) for value in raw_values]

    if task == "numeric":
        return AttributeLabels(
            attribute_name=attribute_name,
            task=task,
            y=np.asarray(raw_values, dtype=float),
            labels=labels,
        )

    if task == "ordinal":
        ordered = [
            str(value)
            for value in persona_dataset.attribute_info(attribute_name)[
                "ordered_values"
            ]
        ]
        ranks = {value: idx for idx, value in enumerate(ordered)}
        return AttributeLabels(
            attribute_name=attribute_name,
            task=task,
            y=np.asarray([ranks[value] for value in labels], dtype=int),
            labels=labels,
            class_names=ordered,
        )

    unique = sorted(set(labels))
    if task == "binary":
        if len(unique) != 2:
            raise ValueError(
                f"{attribute_name!r} must have exactly two observed values; got {unique}"
            )
        negative, positive = unique
        y = np.asarray([1 if value == positive else 0 for value in labels], dtype=int)
        return AttributeLabels(
            attribute_name=attribute_name,
            task=task,
            y=y,
            labels=labels,
            class_names=[negative, positive],
        )

    class_to_idx = {value: idx for idx, value in enumerate(unique)}
    return AttributeLabels(
        attribute_name=attribute_name,
        task="categorical",
        y=np.asarray([class_to_idx[value] for value in labels], dtype=int),
        labels=labels,
        class_names=unique,
    )


# ---------------------------------------------------------------------------
# Probe construction
# ---------------------------------------------------------------------------


def difference_of_means_direction(
    X_train: np.ndarray, y_train: np.ndarray
) -> tuple[np.ndarray, float]:
    """mean(positive) - mean(negative), with the midpoint as a bias."""
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=int)
    neg = X_train[y_train == 0]
    pos = X_train[y_train == 1]
    if len(neg) == 0 or len(pos) == 0:
        raise ValueError("difference_of_means requires both binary classes")

    direction = pos.mean(axis=0) - neg.mean(axis=0)
    midpoint = 0.5 * (neg.mean(axis=0) @ direction + pos.mean(axis=0) @ direction)
    return direction, -float(midpoint)


def predict_difference_of_means(
    X: np.ndarray, direction: np.ndarray, bias: float
) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(X, dtype=np.float32) @ direction + bias
    return (scores >= 0).astype(int), scores


def make_linear_probe(
    probe_kind: ProbeKind,
    n_pca_components: int | None = None,
    seed: int = 0,
) -> Pipeline:
    """Pipeline: scaler (+ optional PCA) + classifier/regressor.

    When ``n_pca_components`` is set, PCA is a pipeline step so it is fit on
    the train split only -- no leakage from the held-out test split.
    """
    if probe_kind == "difference_of_means":
        raise ValueError("difference_of_means is not an sklearn pipeline")

    steps: list = [("scale", StandardScaler())]
    if n_pca_components is not None:
        steps.append(("pca", PCA(n_components=n_pca_components, random_state=seed)))

    if probe_kind == "logistic_regression":
        probe = LogisticRegression(
            class_weight="balanced", max_iter=5000, random_state=seed
        )
    elif probe_kind == "ridge_regression":
        probe = Ridge(alpha=1.0)
    else:
        raise ValueError(f"Unsupported probe kind: {probe_kind}")
    steps.append(("probe", probe))
    return Pipeline(steps)


def default_probe_kinds(task: ProbeTask) -> list[ProbeKind]:
    """Probe kinds run for ``task``. Binary keeps both linear baselines."""
    if task == "binary":
        return ["difference_of_means", "logistic_regression"]
    if task == "categorical":
        return ["logistic_regression"]
    return ["ridge_regression"]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def classification_baseline_metrics(y: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    majority = int(np.bincount(y).argmax())
    predictions = np.full_like(y, majority)
    return {
        "baseline_balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
    }


def regression_baseline_metrics(y: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    predictions = np.full_like(y, float(y.mean()))
    return {
        "baseline_mae": float(mean_absolute_error(y, predictions)),
        "baseline_r2": float(r2_score(y, predictions)),
    }


# ---------------------------------------------------------------------------
# Train/test split + evaluation
# ---------------------------------------------------------------------------


def _split_indices(
    y: np.ndarray,
    *,
    classification: bool,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Single 80/20 split (``random_state=0``).

    Classification tasks stratify on ``y`` so the held-out set preserves the
    attribute balance.
    """
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=TEST_SIZE,
        random_state=seed,
        stratify=y if classification else None,
    )
    return train_idx, test_idx


def _nearest_observed_labels(values: np.ndarray, observed: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    observed = np.unique(np.asarray(observed, dtype=int))
    nearest = np.abs(values[:, None] - observed[None, :]).argmin(axis=1)
    return observed[nearest]


def evaluate_classification(
    X: np.ndarray,
    y: np.ndarray,
    *,
    task: ProbeTask,
    layer: int,
    probe_kind: ProbeKind,
    n_pca_components: int | None = None,
    seed: int = 0,
) -> dict[str, object]:
    """Single 80/20 stratified split for binary/categorical/ordinal labels.

    Ordinal labels are fit as floats (ranks) and test scores are rounded back
    to integer ranks, so balanced_accuracy carries over; MAE on ranks is also
    returned so "how far off" stays visible.
    """
    if task not in CLASSIFICATION_TASKS:
        raise ValueError(f"task must be one of {CLASSIFICATION_TASKS}, got {task!r}")
    if task == "ordinal" and probe_kind != "ridge_regression":
        raise ValueError("ordinal probing uses ridge_regression")
    if task != "ordinal" and probe_kind == "ridge_regression":
        raise ValueError("ridge_regression is only used for ordinal/numeric tasks")

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=int)
    observed_labels = np.unique(y)
    if probe_kind == "difference_of_means" and len(observed_labels) != 2:
        raise ValueError("difference_of_means only supports binary labels")

    train_idx, test_idx = _split_indices(y, classification=True, seed=seed)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    if probe_kind == "difference_of_means":
        X_train_f, X_test_f = _maybe_pca(X_train, X_test, n_pca_components, seed)
        direction, bias = difference_of_means_direction(X_train_f, y_train)
        predictions, _ = predict_difference_of_means(X_test_f, direction, bias)
    else:
        pipeline = make_linear_probe(probe_kind, n_pca_components, seed=seed)
        pipeline.fit(
            X_train, y_train.astype(float) if task == "ordinal" else y_train
        )
        if task == "ordinal":
            low, high = int(y.min()), int(y.max())
            rounded = np.clip(np.rint(pipeline.predict(X_test)), low, high)
            predictions = _nearest_observed_labels(rounded, observed_labels)
        else:
            predictions = pipeline.predict(X_test)

    mae = (
        float(mean_absolute_error(y_test, predictions))
        if task == "ordinal"
        else None
    )
    return {
        "layer": layer,
        "probe_kind": probe_kind,
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "mae": mae,
        "r2": None,
        **classification_baseline_metrics(y_test),
    }


def evaluate_regression(
    X: np.ndarray,
    y: np.ndarray,
    *,
    layer: int,
    probe_kind: ProbeKind = "ridge_regression",
    n_pca_components: int | None = None,
    seed: int = 0,
) -> dict[str, object]:
    """Single 80/20 split for numeric attributes (ridge regression)."""
    if probe_kind != "ridge_regression":
        raise ValueError("numeric probing uses ridge_regression")
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=float)

    train_idx, test_idx = _split_indices(y, classification=False, seed=seed)
    pipeline = make_linear_probe(probe_kind, n_pca_components, seed=seed)
    pipeline.fit(X[train_idx], y[train_idx])
    predictions = pipeline.predict(X[test_idx])
    y_test = y[test_idx]

    return {
        "layer": layer,
        "probe_kind": probe_kind,
        "balanced_accuracy": None,
        "mae": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
        **regression_baseline_metrics(y_test),
    }


def _maybe_pca(
    X_train: np.ndarray, X_test: np.ndarray, n_pca_components: int | None, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """PCA the split (fit on train only) when ``n_pca_components`` is set."""
    if n_pca_components is None:
        return X_train, X_test
    pca = PCA(n_components=n_pca_components, random_state=seed)
    return pca.fit_transform(X_train), pca.transform(X_test)


# ---------------------------------------------------------------------------
# Sweeps and controls
# ---------------------------------------------------------------------------


def sweep_attribute(
    samples: LayeredSamples,
    labels: AttributeLabels,
    *,
    layers: list[int],
    probe_kinds: list[ProbeKind] | None = None,
    n_pca_components: int | None = None,
    seed: int = 0,
) -> list[dict[str, object]]:
    """Sweep one attribute across layers x probe_kinds; return metric rows.

    Auto-dispatches by ``labels.task`` unless ``probe_kinds`` is passed. Set
    ``n_pca_components`` to fit a per-split PCA (e.g. 10) instead of raw
    activations.
    """
    probe_kinds = probe_kinds or default_probe_kinds(labels.task)
    vectors = samples.vectors.float().cpu().numpy()

    rows: list[dict[str, object]] = []
    for layer in layers:
        X = vectors[:, layer, :]
        for probe_kind in probe_kinds:
            if labels.task == "numeric":
                metrics = evaluate_regression(
                    X,
                    labels.y,
                    layer=layer,
                    probe_kind=probe_kind,
                    n_pca_components=n_pca_components,
                    seed=seed,
                )
            else:
                metrics = evaluate_classification(
                    X,
                    labels.y,
                    task=labels.task,
                    layer=layer,
                    probe_kind=probe_kind,
                    n_pca_components=n_pca_components,
                    seed=seed,
                )
            rows.append(_metric_row(labels, metrics))
    return rows


def shuffle_label_baseline(
    X: np.ndarray,
    y: np.ndarray,
    *,
    task: ProbeTask,
    layer: int,
    probe_kind: ProbeKind,
    n_pca_components: int | None = None,
    n_repeats: int = 5,
    seed: int = 0,
) -> dict[str, float]:
    """Selectivity control: train the same probe on shuffled labels.

    A probe that does well on shuffled labels is memorizing dataset
    artifacts, not reading out the property. The gap between the real-label
    metric and this shuffled metric is the *selectivity* (Hewitt & Liang 2019).
    Classification tasks only.
    """
    if task not in CLASSIFICATION_TASKS:
        raise ValueError(
            "shuffle_label_baseline is only defined for classification tasks"
        )
    rng = np.random.default_rng(seed)
    accuracies: list[float] = []
    for repeat in range(n_repeats):
        shuffled = rng.permutation(y)
        metrics = evaluate_classification(
            X,
            shuffled,
            task=task,
            layer=layer,
            probe_kind=probe_kind,
            n_pca_components=n_pca_components,
            seed=seed + repeat,
        )
        accuracies.append(float(metrics["balanced_accuracy"]))

    balanced = np.asarray(accuracies)
    return {
        "balanced_accuracy_mean": float(balanced.mean()),
        "balanced_accuracy_std": float(balanced.std()),
    }


def filter_attribute_samples_min_count(
    samples: LayeredSamples,
    labels: AttributeLabels,
    min_count: int,
) -> tuple[LayeredSamples, AttributeLabels]:
    """Drop categorical/ordinal classes too rare for a stratified split.

    Ordinal ranks are preserved (rank ordering still matters); for
    binary/categorical, surviving classes are renumbered to 0..k-1.
    """
    if labels.task not in {"binary", "categorical", "ordinal"}:
        return samples, labels
    y = np.asarray(labels.y, dtype=int)
    counts = np.bincount(y)
    keep_classes = {idx for idx, count in enumerate(counts) if count >= min_count}
    keep_mask = np.asarray([value in keep_classes for value in y], dtype=bool)
    if keep_mask.all():
        return samples, labels

    keep_indices = np.flatnonzero(keep_mask).tolist()
    class_names = labels.class_names
    if labels.task == "ordinal":
        filtered_y = y[keep_mask]
        filtered_class_names = class_names
    else:
        old_to_new = {old: new for new, old in enumerate(sorted(keep_classes))}
        filtered_y = np.asarray(
            [old_to_new[int(value)] for value in y[keep_mask]], dtype=int
        )
        filtered_class_names = (
            [class_names[old] for old in sorted(keep_classes)]
            if class_names is not None
            else None
        )
    filtered_samples = LayeredSamples(
        vectors=samples.vectors[keep_indices],
        labels=[samples.labels[idx] for idx in keep_indices],
        hover_text=[samples.hover_text[idx] for idx in keep_indices],
    )
    return filtered_samples, AttributeLabels(
        attribute_name=labels.attribute_name,
        task=labels.task,
        y=filtered_y,
        labels=[labels.labels[idx] for idx in keep_indices],
        class_names=filtered_class_names,
    )


def _metric_row(
    labels: AttributeLabels,
    metrics: dict[str, object],
) -> dict[str, object]:
    return {
        "attribute": labels.attribute_name,
        "layer": metrics["layer"],
        "probe_kind": metrics["probe_kind"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "mae": metrics["mae"],
        "r2": metrics["r2"],
        **{
            key: value
            for key, value in metrics.items()
            if key.startswith("baseline_")
        },
    }


# ---------------------------------------------------------------------------
# Final-fit artifact saving (persona-ui compatible)
# ---------------------------------------------------------------------------


def _scaler_tensors(scaler: StandardScaler) -> dict[str, torch.Tensor]:
    return {
        "scaler_mean": torch.from_numpy(scaler.mean_.astype(np.float32)),
        "scaler_scale": torch.from_numpy(scaler.scale_.astype(np.float32)),
    }


def _pca_tensors(pca: PCA) -> dict[str, torch.Tensor]:
    # sklearn's components_ is a non-contiguous view; safetensors needs C-order.
    return {
        "pca_mean": torch.from_numpy(
            np.ascontiguousarray(pca.mean_, dtype=np.float32)
        ),
        "pca_components": torch.from_numpy(
            np.ascontiguousarray(pca.components_, dtype=np.float32)
        ),
    }


def _fit_difference_artifact(
    X: np.ndarray, y: np.ndarray, n_pca_components: int | None, seed: int
) -> dict[str, torch.Tensor]:
    if n_pca_components is not None:
        scaler = StandardScaler()
        pca = PCA(n_components=n_pca_components, random_state=seed)
        features = pca.fit_transform(scaler.fit_transform(X))
        transform_tensors = {**_scaler_tensors(scaler), **_pca_tensors(pca)}
    else:
        features = X
        transform_tensors = {}

    direction, bias = difference_of_means_direction(features, y)
    # Two-row weight matrix so persona-ui's 2-class linear head loads it directly.
    weights = np.stack([-0.5 * direction, 0.5 * direction]).astype(np.float32)
    biases = np.asarray([-0.5 * bias, 0.5 * bias], dtype=np.float32)
    return {
        "weight": torch.from_numpy(weights),
        "bias": torch.from_numpy(biases),
        "direction": torch.from_numpy(direction.astype(np.float32)),
        "direction_bias": torch.tensor([bias], dtype=torch.float32),
        **transform_tensors,
    }


def _fit_pipeline_artifact(
    X: np.ndarray,
    y: np.ndarray,
    task: ProbeTask,
    probe_kind: ProbeKind,
    n_pca_components: int | None,
    seed: int,
) -> dict[str, torch.Tensor]:
    pipeline = make_linear_probe(probe_kind, n_pca_components, seed=seed)
    fit_y = y.astype(float) if task in {"ordinal", "numeric"} else y
    pipeline.fit(X, fit_y)

    scaler = pipeline.named_steps["scale"]
    final = pipeline.named_steps["probe"]
    tensors: dict[str, torch.Tensor] = dict(_scaler_tensors(scaler))
    if "pca" in pipeline.named_steps:
        tensors.update(_pca_tensors(pipeline.named_steps["pca"]))

    coef = np.asarray(final.coef_, dtype=np.float32)
    intercept = np.asarray(
        getattr(final, "intercept_", np.zeros(coef.shape[0])), dtype=np.float32
    )
    if coef.ndim == 1:
        coef = coef.reshape(1, -1)
    if intercept.ndim == 0:
        intercept = intercept.reshape(1)

    if task == "binary" and coef.shape[0] == 1:
        # Same 2-row layout as the diff-of-means artifact, for UI consistency.
        weight = np.vstack([-0.5 * coef[0], 0.5 * coef[0]]).astype(np.float32)
        bias = np.asarray([-0.5 * intercept[0], 0.5 * intercept[0]], dtype=np.float32)
    else:
        weight = coef.astype(np.float32)
        bias = intercept.astype(np.float32)
    tensors.update(
        {
            "weight": torch.from_numpy(weight),
            "bias": torch.from_numpy(bias),
        }
    )
    return tensors


def _json_safe(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def save_probe_artifact(
    *,
    X: np.ndarray,
    y: np.ndarray,
    labels: AttributeLabels,
    task: ProbeTask,
    probe_kind: ProbeKind,
    layer: int,
    model_name: str,
    variant: str,
    mask_strategy: object,
    n_pca_components: int | None = None,
    output_dir: str | Path = "artifacts/probes",
    metrics: dict[str, object] | None = None,
    seed: int = 0,
    location: str = "post_reasoning",
) -> Path:
    """Refit on all personas and save a lightweight artifact tree.

    Writes:
    - ``probe.json`` -- schema metadata, including the eval metrics passed in.
    - ``weights.safetensors`` -- portable tensor bundle (scaler, optional PCA,
      weight, bias, plus the diff-of-means direction when applicable).

    Returns the artifact directory.
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    if probe_kind == "difference_of_means":
        tensors = _fit_difference_artifact(
            X, y.astype(int), n_pca_components, seed
        )
    else:
        tensors = _fit_pipeline_artifact(
            X, y, task, probe_kind, n_pca_components, seed
        )

    suffix = "" if n_pca_components is None else f"_pca{n_pca_components}"
    root = (
        Path(output_dir)
        / model_dir_name(model_name)
        / normalize_mask_strategy(mask_strategy)
        / variant
        / labels.attribute_name
        / f"{probe_kind}{suffix}_layer{layer}"
    )
    root.mkdir(parents=True, exist_ok=True)
    metadata_path = root / "probe.json"
    weights_path = root / "weights.safetensors"

    metadata = {
        "schema_version": 2,
        "model_name": model_name,
        "variant": variant,
        "mask_strategy": normalize_mask_strategy(mask_strategy),
        "attribute_name": labels.attribute_name,
        "task": task,
        "probe_kind": probe_kind,
        "n_pca_components": n_pca_components,
        "layer": layer,
        "location": location,
        "input_dim": int(X.shape[1]),
        "artifact_feature_dim": int(tensors["weight"].shape[1]),
        "class_names": labels.class_names,
        "metrics": metrics or {},
    }
    metadata_path.write_text(json.dumps(_json_safe(metadata), indent=2))
    save_file(tensors, str(weights_path))
    return root


def layer_matrix(samples: LayeredSamples, layer: int) -> np.ndarray:
    """Return a (n_personas, hidden_size) numpy matrix for one layer."""
    return samples.vectors[:, layer, :].float().cpu().numpy()


def pick_layers(num_layers: int, fast: bool = True) -> list[int]:
    """Five evenly-spaced layers when ``fast``, otherwise every layer."""
    if not fast:
        return list(range(num_layers))
    return sorted(
        {0, num_layers // 4, num_layers // 2, (3 * num_layers) // 4, num_layers - 1}
    )


def best_row(
    rows: list[dict[str, object]],
    metric: str,
    *,
    higher_is_better: bool = True,
) -> dict[str, object]:
    """Pick the row that wins on ``metric`` (rows with NaN/None on ``metric`` are skipped)."""
    valid = [
        row
        for row in rows
        if row.get(metric) is not None and np.isfinite(float(row[metric]))
    ]
    if not valid:
        raise ValueError(f"No rows with finite {metric!r}")
    return max(
        valid, key=lambda row: float(row[metric]) * (1 if higher_is_better else -1)
    )


def primary_metric(task: ProbeTask) -> str:
    """Return the metric used to pick the best probe for ``task``."""
    return "r2" if task == "numeric" else "balanced_accuracy"


def run_attribute_probe(
    samples: LayeredSamples,
    persona_dataset,
    attribute: str,
    persona_ids: list[str],
    *,
    layers: list[int],
    n_pca_components: int | None = None,
    min_class_count: int = 5,
    model_name: str,
    variant: str,
    mask_strategy: object,
    output_dir: str | Path,
    seed: int = 0,
) -> tuple[Path, dict[str, object], ProbeTask]:
    """Sweep one attribute, refit the best config, and save the artifact.

    Returns ``(directory, best_row, task)`` where ``best_row`` is the winning
    sweep row (handy for CLI summaries) and ``task`` is the inferred task.
    """
    task = infer_probe_task(persona_dataset, attribute)
    labels = attribute_probe_labels(persona_dataset, attribute, persona_ids, task=task)
    probe_samples, labels = filter_attribute_samples_min_count(
        samples,
        labels,
        min_count=min_class_count,
    )
    rows = sweep_attribute(
        probe_samples,
        labels,
        layers=layers,
        n_pca_components=n_pca_components,
        seed=seed,
    )
    best = best_row(rows, primary_metric(task))
    layer = int(best["layer"])
    probe_kind = str(best["probe_kind"])
    directory = save_probe_artifact(
        X=layer_matrix(probe_samples, layer),
        y=labels.y,
        labels=labels,
        task=task,
        probe_kind=probe_kind,  # type: ignore[arg-type]
        layer=layer,
        model_name=model_name,
        variant=variant,
        mask_strategy=mask_strategy,
        n_pca_components=n_pca_components,
        output_dir=output_dir,
        metrics=best,
        seed=seed,
    )
    return directory, best, task
