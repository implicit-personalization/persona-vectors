"""Linear probes over persona vectors.

Three probe kinds,:

- ``difference_of_means``: the Anthropic-style persona-vector direction
  (mean(positive) - mean(negative)), with a midpoint bias. Closed-form,
  most interpretable. Binary only.

- ``logistic_regression``: the canonical probing classifier. Class-balanced,
  L2-regularized, with a StandardScaler. Handles binary and multi-class.

- ``ridge_regression``: linear regression for ordinal ranks and numeric
  attributes. Ordinal predictions are rounded back to the rank scale.

Evaluation is 5-fold cross-validation by default (StratifiedKFold for
classification, KFold for regression). The scaler and any optional PCA
are fit *inside each fold's pipeline*, so there is no leakage from the
held-out personas. Final saved artifacts are refit on all personas.
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
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from persona_vectors.analysis import LayeredSamples
from persona_vectors.artifacts import model_dir_name, normalize_mask_strategy

ProbeTask = Literal["binary", "ordinal", "categorical", "numeric"]
ProbeKind = Literal["difference_of_means", "logistic_regression", "ridge_regression"]
FeatureSpace = Literal["raw", "pca10"]

CLASSIFICATION_TASKS: frozenset[ProbeTask] = frozenset(
    {"binary", "categorical", "ordinal"}
)


@dataclass(frozen=True)
class AttributeLabels:
    """Stable label container for one persona attribute."""

    attribute_name: str
    task: ProbeTask
    y: np.ndarray
    labels: list[str]
    class_names: list[str] | None = None


@dataclass(frozen=True)
class ProbeMetrics:
    """Out-of-fold CV metrics for one probe configuration.

    Each task keeps only the metric that matters: balanced_accuracy for
    classification (binary/categorical/ordinal), plus mae for ordinal rank
    distance; r2 + mae for numeric.
    """

    layer: int
    probe_kind: str
    feature_space: str
    balanced_accuracy: float | None = None
    mae: float | None = None
    r2: float | None = None


@dataclass(frozen=True)
class ProbeCV:
    """Per-fold predictions assembled into a single out-of-fold array."""

    metrics: ProbeMetrics
    predictions: np.ndarray
    scores: np.ndarray


@dataclass(frozen=True)
class ProbeSweep:
    rows: list[dict[str, object]]
    predictions: dict[tuple[str, str, int], np.ndarray]
    scores: dict[tuple[str, str, int], np.ndarray]


@dataclass(frozen=True)
class SavedProbeArtifact:
    directory: Path
    metadata_path: Path
    weights_path: Path
    pt_path: Path | None


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
    feature_space: FeatureSpace,
    seed: int = 0,
) -> Pipeline:
    """Build a fold-local pipeline. Scaler + optional PCA + classifier/regressor.

    PCA (when requested) is fit inside the pipeline, so each CV fold gets its
    own PCA fit on training personas only -- no leakage from held-out folds.
    """
    if probe_kind == "difference_of_means":
        raise ValueError("difference_of_means is not an sklearn pipeline")

    steps: list = [("scale", StandardScaler())]
    if feature_space == "pca10":
        steps.append(("pca", PCA(n_components=10, random_state=seed)))
    elif feature_space != "raw":
        raise ValueError("feature_space must be 'raw' or 'pca10'")

    if probe_kind == "logistic_regression":
        probe = LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=seed
        )
    elif probe_kind == "ridge_regression":
        probe = Ridge(alpha=1.0)
    else:
        raise ValueError(f"Unsupported probe kind: {probe_kind}")
    steps.append(("probe", probe))
    return Pipeline(steps)


def default_probe_kinds(task: ProbeTask) -> list[ProbeKind]:
    """Default probe kinds per task. Binary keeps both linear baselines."""
    if task == "binary":
        return ["difference_of_means", "logistic_regression"]
    if task == "categorical":
        return ["logistic_regression"]
    return ["ridge_regression"]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _probe_scores(pipeline: Pipeline, X: np.ndarray) -> np.ndarray:
    """Return a 1-D score for binary, otherwise a (n, n_classes) matrix."""
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(X)
        return np.asarray(proba[:, 1] if proba.shape[1] == 2 else proba, dtype=float)
    if hasattr(pipeline, "decision_function"):
        return np.asarray(pipeline.decision_function(X), dtype=float)
    return np.asarray(pipeline.predict(X), dtype=float)


def _classification_metrics(
    y: np.ndarray,
    predictions: np.ndarray,
    *,
    layer: int,
    probe_kind: str,
    feature_space: str,
    mae: float | None = None,
) -> ProbeMetrics:
    return ProbeMetrics(
        layer=layer,
        probe_kind=probe_kind,
        feature_space=feature_space,
        balanced_accuracy=float(balanced_accuracy_score(y, predictions)),
        mae=mae,
    )


def _regression_metrics(
    y: np.ndarray,
    predictions: np.ndarray,
    *,
    layer: int,
    probe_kind: str,
    feature_space: str,
) -> ProbeMetrics:
    return ProbeMetrics(
        layer=layer,
        probe_kind=probe_kind,
        feature_space=feature_space,
        mae=float(mean_absolute_error(y, predictions)),
        r2=float(r2_score(y, predictions)),
    )


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
# Cross-validation
# ---------------------------------------------------------------------------


def _stratified_n_splits(y: np.ndarray, n_splits: int) -> int:
    counts = np.bincount(y)
    min_class_count = int(counts[counts > 0].min())
    return min(n_splits, min_class_count)


def _nearest_observed_labels(values: np.ndarray, observed: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    observed = np.unique(np.asarray(observed, dtype=int))
    nearest = np.abs(values[:, None] - observed[None, :]).argmin(axis=1)
    return observed[nearest]


def cross_validate_classification(
    X: np.ndarray,
    y: np.ndarray,
    *,
    task: ProbeTask,
    layer: int,
    probe_kind: ProbeKind,
    feature_space: FeatureSpace = "raw",
    n_splits: int = 5,
    seed: int = 0,
) -> ProbeCV:
    """5-fold StratifiedKFold for binary, categorical, or ordinal labels.

    Ordinal labels are fit as floats (ranks) and held-out scores are rounded
    back to integer ranks, so balanced_accuracy carries over. MAE on ranks
    is also returned for ordinal so "how far off" stays visible.
    """
    if task not in CLASSIFICATION_TASKS:
        raise ValueError(f"task must be one of {CLASSIFICATION_TASKS}, got {task!r}")
    if task == "ordinal" and probe_kind != "ridge_regression":
        raise ValueError("ordinal probing uses ridge_regression")
    if task != "ordinal" and probe_kind == "ridge_regression":
        raise ValueError("ridge_regression is only used for ordinal/numeric tasks")

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=int)
    n_splits = _stratified_n_splits(y, n_splits)
    if n_splits < 2:
        raise ValueError("Need at least two examples per class for stratified CV")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    predictions = np.zeros_like(y)
    scores: np.ndarray | None = None
    low, high = int(y.min()), int(y.max())
    observed_labels = np.unique(y)
    if probe_kind == "difference_of_means" and len(observed_labels) != 2:
        raise ValueError("difference_of_means only supports binary labels")

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]

        if probe_kind == "difference_of_means":
            X_train_f, X_test_f = _maybe_pca_per_fold(
                X_train, X_test, feature_space, seed
            )
            direction, bias = difference_of_means_direction(X_train_f, y_train)
            fold_pred, fold_score = predict_difference_of_means(
                X_test_f, direction, bias
            )
        else:
            pipeline = make_linear_probe(probe_kind, feature_space, seed=seed)
            pipeline.fit(
                X_train, y_train.astype(float) if task == "ordinal" else y_train
            )
            fold_score = _probe_scores(pipeline, X_test)
            if task == "ordinal":
                rounded = np.clip(np.rint(fold_score), low, high)
                fold_pred = _nearest_observed_labels(rounded, observed_labels)
            else:
                fold_pred = pipeline.predict(X_test)

        predictions[test_idx] = fold_pred
        scores = _scatter_fold_scores(scores, test_idx, fold_score, n=len(y))

    assert scores is not None
    mae = float(mean_absolute_error(y, predictions)) if task == "ordinal" else None
    metrics = _classification_metrics(
        y,
        predictions,
        layer=layer,
        probe_kind=probe_kind,
        feature_space=feature_space,
        mae=mae,
    )
    return ProbeCV(metrics=metrics, predictions=predictions, scores=scores)


def cross_validate_regression(
    X: np.ndarray,
    y: np.ndarray,
    *,
    layer: int,
    probe_kind: ProbeKind = "ridge_regression",
    feature_space: FeatureSpace = "raw",
    n_splits: int = 5,
    seed: int = 0,
) -> ProbeCV:
    """K-fold CV for numeric attributes (ridge regression)."""
    if probe_kind != "ridge_regression":
        raise ValueError("numeric probing uses ridge_regression")
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=float)
    predictions = np.zeros(len(y), dtype=float)
    cv = KFold(n_splits=min(n_splits, len(y)), shuffle=True, random_state=seed)
    for train_idx, test_idx in cv.split(X):
        pipeline = make_linear_probe(probe_kind, feature_space, seed=seed)
        pipeline.fit(X[train_idx], y[train_idx])
        predictions[test_idx] = pipeline.predict(X[test_idx])
    return ProbeCV(
        metrics=_regression_metrics(
            y,
            predictions,
            layer=layer,
            probe_kind=probe_kind,
            feature_space=feature_space,
        ),
        predictions=predictions,
        scores=predictions,
    )


def _maybe_pca_per_fold(
    X_train: np.ndarray, X_test: np.ndarray, feature_space: FeatureSpace, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Fit PCA on training fold only when feature_space=='pca10'."""
    if feature_space == "raw":
        return X_train, X_test
    if feature_space != "pca10":
        raise ValueError("feature_space must be 'raw' or 'pca10'")
    pca = PCA(n_components=10, random_state=seed)
    return pca.fit_transform(X_train), pca.transform(X_test)


def _scatter_fold_scores(
    accumulator: np.ndarray | None,
    test_idx: np.ndarray,
    fold_scores: np.ndarray,
    *,
    n: int,
) -> np.ndarray:
    fold_scores = np.asarray(fold_scores, dtype=float)
    if accumulator is None:
        accumulator = np.zeros((n, *fold_scores.shape[1:]), dtype=float)
    accumulator[test_idx] = fold_scores
    return accumulator


# ---------------------------------------------------------------------------
# Sweeps and controls
# ---------------------------------------------------------------------------


def sweep_attribute(
    samples: LayeredSamples,
    labels: AttributeLabels,
    *,
    layers: list[int],
    probe_kinds: list[ProbeKind] | None = None,
    feature_spaces: list[FeatureSpace] | None = None,
    n_splits: int = 5,
    seed: int = 0,
) -> ProbeSweep:
    """Sweep one attribute across layers x probe_kinds x feature_spaces.

    Auto-dispatches by ``labels.task``. Defaults: ``probe_kinds`` per task
    (see :func:`default_probe_kinds`), ``feature_spaces=['raw']``.
    """
    probe_kinds = probe_kinds or default_probe_kinds(labels.task)
    feature_spaces = feature_spaces or ["raw"]
    vectors = samples.vectors.float().cpu().numpy()

    if labels.task == "numeric":
        baseline = regression_baseline_metrics(labels.y)
    else:
        baseline = classification_baseline_metrics(labels.y)

    rows: list[dict[str, object]] = []
    predictions: dict[tuple[str, str, int], np.ndarray] = {}
    scores: dict[tuple[str, str, int], np.ndarray] = {}

    for layer in layers:
        X = vectors[:, layer, :]
        for probe_kind in probe_kinds:
            for feature_space in feature_spaces:
                if labels.task == "numeric":
                    result = cross_validate_regression(
                        X,
                        labels.y,
                        layer=layer,
                        probe_kind=probe_kind,
                        feature_space=feature_space,
                        n_splits=n_splits,
                        seed=seed,
                    )
                else:
                    result = cross_validate_classification(
                        X,
                        labels.y,
                        task=labels.task,
                        layer=layer,
                        probe_kind=probe_kind,
                        feature_space=feature_space,
                        n_splits=n_splits,
                        seed=seed,
                    )
                key = (probe_kind, feature_space, layer)
                predictions[key] = result.predictions
                scores[key] = result.scores
                rows.append(_metric_row(labels, result, baseline))
    return ProbeSweep(rows=rows, predictions=predictions, scores=scores)


def shuffle_label_baseline(
    X: np.ndarray,
    y: np.ndarray,
    *,
    task: ProbeTask,
    layer: int,
    probe_kind: ProbeKind,
    feature_space: FeatureSpace = "raw",
    n_repeats: int = 5,
    n_splits: int = 5,
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
    metrics_per_run: list[ProbeMetrics] = []
    for repeat in range(n_repeats):
        shuffled = rng.permutation(y)
        result = cross_validate_classification(
            X,
            shuffled,
            task=task,
            layer=layer,
            probe_kind=probe_kind,
            feature_space=feature_space,
            n_splits=n_splits,
            seed=seed + repeat,
        )
        metrics_per_run.append(result.metrics)

    balanced = np.asarray([m.balanced_accuracy for m in metrics_per_run])
    return {
        "balanced_accuracy_mean": float(balanced.mean()),
        "balanced_accuracy_std": float(balanced.std()),
    }


def filter_attribute_samples_min_count(
    samples: LayeredSamples,
    labels: AttributeLabels,
    min_count: int,
) -> tuple[LayeredSamples, AttributeLabels]:
    """Drop categorical/ordinal classes too rare for stratified CV.

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
    result: ProbeCV,
    baseline: dict[str, float],
) -> dict[str, object]:
    return {
        "attribute": labels.attribute_name,
        "layer": result.metrics.layer,
        "probe_kind": result.metrics.probe_kind,
        "feature_space": result.metrics.feature_space,
        "balanced_accuracy": result.metrics.balanced_accuracy,
        "mae": result.metrics.mae,
        "r2": result.metrics.r2,
        **baseline,
    }


# ---------------------------------------------------------------------------
# Final-fit artifact saving (persona-ui compatible)
# ---------------------------------------------------------------------------


def _fit_final_pipeline(
    X: np.ndarray,
    y: np.ndarray,
    *,
    task: ProbeTask,
    probe_kind: ProbeKind,
    feature_space: FeatureSpace,
    seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    """Refit the chosen probe on all personas and return its persisted tensors."""
    if probe_kind == "difference_of_means":
        return _fit_difference_artifact(X, y.astype(int), feature_space, seed)
    return _fit_pipeline_artifact(X, y, task, probe_kind, feature_space, seed)


def _scaler_tensors(scaler: StandardScaler) -> dict[str, torch.Tensor]:
    return {
        "scaler_mean": torch.from_numpy(scaler.mean_.astype(np.float32)),
        "scaler_scale": torch.from_numpy(scaler.scale_.astype(np.float32)),
    }


def _pca_tensors(pca: PCA) -> dict[str, torch.Tensor]:
    return {
        "pca_mean": torch.from_numpy(pca.mean_.astype(np.float32)),
        "pca_components": torch.from_numpy(pca.components_.astype(np.float32)),
    }


def _fit_difference_artifact(X, y, feature_space, seed):
    if feature_space == "pca10":
        scaler = StandardScaler()
        pca = PCA(n_components=10, random_state=seed)
        features = pca.fit_transform(scaler.fit_transform(X))
        transform_tensors = {**_scaler_tensors(scaler), **_pca_tensors(pca)}
        transform_metadata = {"pca_components": 10}
    else:
        features = X
        transform_tensors = {}
        transform_metadata = {}

    direction, bias = difference_of_means_direction(features, y)
    # Two-row weight matrix so persona-ui's 2-class linear head loads it directly.
    weights = np.stack([-0.5 * direction, 0.5 * direction]).astype(np.float32)
    biases = np.asarray([-0.5 * bias, 0.5 * bias], dtype=np.float32)
    tensors = {
        "weight": torch.from_numpy(weights),
        "bias": torch.from_numpy(biases),
        "direction": torch.from_numpy(direction.astype(np.float32)),
        "direction_bias": torch.tensor([bias], dtype=torch.float32),
        **transform_tensors,
    }
    return tensors, transform_metadata


def _fit_pipeline_artifact(X, y, task, probe_kind, feature_space, seed):
    pipeline = make_linear_probe(probe_kind, feature_space, seed=seed)
    fit_y = y.astype(float) if task in {"ordinal", "numeric"} else y
    pipeline.fit(X, fit_y)

    scaler = pipeline.named_steps["scale"]
    final = pipeline.named_steps["probe"]
    tensors: dict[str, torch.Tensor] = dict(_scaler_tensors(scaler))
    transform_metadata: dict[str, object] = {}
    if "pca" in pipeline.named_steps:
        pca = pipeline.named_steps["pca"]
        tensors.update(_pca_tensors(pca))
        transform_metadata["pca_components"] = int(pca.n_components_)

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
    return tensors, transform_metadata


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
    feature_space: FeatureSpace,
    layer: int,
    model_name: str,
    variant: str,
    mask_strategy: object,
    output_dir: str | Path = "artifacts/probes",
    metrics: dict[str, object] | None = None,
    seed: int = 0,
    location: str = "post_reasoning",
) -> SavedProbeArtifact:
    """Refit on all personas and save a lightweight artifact tree.

    Writes:
    - ``probe.json`` -- schema metadata, including the CV metrics passed in.
    - ``weights.safetensors`` -- portable tensor bundle (scaler, optional PCA,
      weight, bias, plus the diff-of-means direction when applicable).
    - ``probe.pt`` (binary & categorical only, raw feature space) -- the
      persona-ui-compatible payload with a 2-row linear head + scaler.
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    tensors, transform_metadata = _fit_final_pipeline(
        X,
        y,
        task=task,
        probe_kind=probe_kind,
        feature_space=feature_space,
        seed=seed,
    )

    root = (
        Path(output_dir)
        / model_dir_name(model_name)
        / normalize_mask_strategy(mask_strategy)
        / variant
        / labels.attribute_name
        / f"{probe_kind}_{feature_space}_layer{layer}"
    )
    root.mkdir(parents=True, exist_ok=True)
    metadata_path = root / "probe.json"
    weights_path = root / "weights.safetensors"

    metadata = {
        "schema_version": 1,
        "model_name": model_name,
        "variant": variant,
        "mask_strategy": normalize_mask_strategy(mask_strategy),
        "attribute_name": labels.attribute_name,
        "task": task,
        "probe_kind": probe_kind,
        "feature_space": feature_space,
        "layer": layer,
        "location": location,
        "input_dim": int(X.shape[1]),
        "artifact_feature_dim": int(tensors["weight"].shape[1]),
        "class_names": labels.class_names,
        "metrics": metrics or {},
        **transform_metadata,
    }
    metadata_path.write_text(json.dumps(_json_safe(metadata), indent=2))
    save_file(tensors, str(weights_path))

    pt_path = None
    if task in {"binary", "categorical"} and feature_space == "raw":
        pt_path = root / "probe.pt"
        torch.save(
            {
                "model_type": "linear",
                "model_state_dict": {
                    "linear.weight": tensors["weight"],
                    "linear.bias": tensors["bias"],
                },
                "input_dim": int(X.shape[1]),
                "num_classes": int(tensors["weight"].shape[0]),
                "idx_to_label": dict(enumerate(labels.class_names or [])),
                "layer": int(layer),
                "location": location,
                "scaler_mean": tensors.get("scaler_mean"),
                "scaler_std": tensors.get("scaler_scale"),
                "model_name": model_name,
                "attribute_name": labels.attribute_name,
                "task": task,
                "probe_kind": probe_kind,
            },
            pt_path,
        )

    return SavedProbeArtifact(
        directory=root,
        metadata_path=metadata_path,
        weights_path=weights_path,
        pt_path=pt_path,
    )


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
    feature_spaces: list[FeatureSpace],
    n_splits: int = 5,
    min_class_count: int = 5,
    model_name: str,
    variant: str,
    mask_strategy: object,
    output_dir: str | Path,
    seed: int = 0,
) -> tuple[SavedProbeArtifact, dict[str, object], ProbeTask]:
    """Sweep one attribute, refit the best config, and save the artifact.

    Returns ``(artifact, best_row, task)`` where ``best_row`` is the winning
    sweep row (handy for CLI summaries) and ``task`` is the inferred task.
    """
    task = infer_probe_task(persona_dataset, attribute)
    labels = attribute_probe_labels(persona_dataset, attribute, persona_ids, task=task)
    probe_samples, labels = filter_attribute_samples_min_count(
        samples,
        labels,
        min_count=min_class_count,
    )
    sweep = sweep_attribute(
        probe_samples,
        labels,
        layers=layers,
        feature_spaces=feature_spaces,
        n_splits=n_splits,
        seed=seed,
    )
    best = best_row(sweep.rows, primary_metric(task))
    layer = int(best["layer"])
    probe_kind = str(best["probe_kind"])
    feature_space = str(best["feature_space"])
    artifact = save_probe_artifact(
        X=layer_matrix(probe_samples, layer),
        y=labels.y,
        labels=labels,
        task=task,
        probe_kind=probe_kind,  # type: ignore[arg-type]
        feature_space=feature_space,  # type: ignore[arg-type]
        layer=layer,
        model_name=model_name,
        variant=variant,
        mask_strategy=mask_strategy,
        output_dir=output_dir,
        metrics=best,
        seed=seed,
    )
    return artifact, best, task
