"""Plots for probe sweeps.

Kept separate from the projection/similarity machinery in ``plots`` so the
probe notebooks (and the persona-ui Probing tab) can import a small,
self-contained surface.
"""

from collections.abc import Iterable, Mapping

import numpy as np
import plotly.graph_objects as go
from plotly.colors import qualitative as _qualitative

from ._common import apply_fig_fonts

__all__ = [
    "plot_attribute_layer_selectivity_heatmap",
    "plot_metric_comparison",
    "plot_metric_over_layers",
]

# Metrics in [0, 1] get a fixed y-range; r2/mae are autoscaled.
_AUTOSCALE_METRICS = {"r2", "mae"}
_BASELINE_LINE = dict(color="#64748b", dash="dot", width=2)
_LEGEND = dict(x=0.98, y=0.02, xanchor="right", yanchor="bottom")


def _metric_y_range(metric: str) -> list[float] | None:
    return None if metric in _AUTOSCALE_METRICS else [0.0, 1.02]


def _add_baseline(fig: go.Figure, baseline: float | None) -> None:
    """Draw the majority-class / mean-prediction baseline as a dotted line."""
    if baseline is None:
        return
    fig.add_hline(
        y=baseline,
        line=_BASELINE_LINE,
        annotation_text=f"baseline {baseline:.3f}",
        annotation_position="bottom right",
    )


def _style_line_fig(
    fig: go.Figure, *, title: str, yaxis_title: str, metric: str
) -> go.Figure:
    """Shared layout for the metric line plots (fonts, legend, axes)."""
    fig.update_layout(
        template="plotly_white",
        legend=_LEGEND,
        xaxis_title="Layer",
        yaxis=dict(title=yaxis_title, range=_metric_y_range(metric)),
    )
    return apply_fig_fonts(fig, title=title)


_COMPARISON_DASHES = ("solid", "dash", "dot", "dashdot")


def _metric_lines(
    rows_by_label: Mapping[str, list[dict[str, object]]],
    attributes: list[str],
    metric: str,
    *,
    color_by: str,
) -> go.Figure:
    """Shared engine for the metric line plots.

    ``color_by="probe_kind"`` gives one auto-colored line per probe_kind (the
    single-sweep view). ``color_by="attribute"`` gives one line per
    (attribute, label): a distinct color per attribute and a dash per label,
    for overlaying full vs compressed sweeps across several attributes.
    """
    fig = go.Figure()
    baseline_drawn = False
    for attr_idx, attr in enumerate(attributes):
        attr_color = _qualitative.Plotly[attr_idx % len(_qualitative.Plotly)]
        for dash, (label, rows) in zip(
            _COMPARISON_DASHES, rows_by_label.items(), strict=False
        ):
            attr_rows = [
                row
                for row in rows
                if row["attribute"] == attr and row.get(metric) is not None
            ]
            if not attr_rows:
                continue
            if not baseline_drawn and attr_idx == 0:
                _add_baseline(fig, attr_rows[0].get(f"baseline_{metric}"))
                baseline_drawn = True
            for probe_kind in sorted({r["probe_kind"] for r in attr_rows}):
                series = [r for r in attr_rows if r["probe_kind"] == probe_kind]
                if color_by == "probe_kind":
                    name = str(probe_kind)
                    line = dict(dash=dash)  # color auto-assigned per trace
                else:
                    name = f"{attr} · {label}"
                    line = dict(color=attr_color, dash=dash)
                fig.add_trace(
                    go.Scatter(
                        x=[r["layer"] for r in series],
                        y=[r[metric] for r in series],
                        mode="lines+markers",
                        name=name,
                        line=line,
                        hovertemplate=(
                            f"{name}<br>Layer %{{x}}<br>"
                            f"{metric}: %{{y:.3f}}<extra></extra>"
                        ),
                    )
                )

    label = attributes[0] if len(attributes) == 1 else "Attributes"
    title = f"{label}: {metric} over layers"
    return _style_line_fig(fig, title=title, yaxis_title=metric, metric=metric)


def plot_metric_over_layers(
    rows: list[dict[str, object]],
    attribute_name: str,
    *,
    metric: str = "balanced_accuracy",
) -> go.Figure:
    """Line plot of one metric over layers, one trace per probe_kind.

    Thin single-sweep wrapper over :func:`_metric_lines`. The majority-class /
    mean-prediction baseline, when present, is drawn as a dotted line.
    """
    return _metric_lines({"": rows}, [attribute_name], metric, color_by="probe_kind")


def plot_metric_comparison(
    rows_by_label: Mapping[str, list[dict[str, object]]],
    attribute_name: str | list[str],
    *,
    metric: str = "balanced_accuracy",
) -> go.Figure:
    """Overlay one metric over layers for several labelled sweeps.

    ``rows_by_label`` maps a short label (e.g. ``"full"``, ``"pca10"``) to a
    sweep's rows. One trace per (attribute, label); each attribute gets a
    distinct color and each label gets its own dash style so full vs compressed
    features are easy to distinguish for multiple attributes in one figure.
    """
    attributes = (
        [attribute_name] if isinstance(attribute_name, str) else list(attribute_name)
    )
    return _metric_lines(rows_by_label, attributes, metric, color_by="attribute")


def plot_attribute_layer_selectivity_heatmap(
    rows_by_attribute: Mapping[str, Iterable[dict[str, object]]],
    *,
    metric: str = "balanced_accuracy",
    subtract_baseline: bool = True,
    title: str | None = None,
    colorscale: str = "RdBu",
    zmid: float | None = 0.0,
) -> go.Figure:
    """Attribute × layer heatmap of probe selectivity.

    ``rows_by_attribute`` maps attribute name to sweep rows. For each
    (attribute, layer) cell the best probe_kind is taken; this is the "upper
    bound" view that reviewers usually ask for. Selectivity is
    ``metric - baseline_<metric>`` when ``subtract_baseline`` is true. For
    ``mae``, lower is better, so values are flipped before plotting.

    Pass ``subtract_baseline=False`` and ``zmid=None`` for a raw-metric view.
    """

    attributes = list(rows_by_attribute)
    if not attributes:
        raise ValueError("rows_by_attribute must contain at least one attribute")

    per_attribute: dict[str, dict[int, float]] = {}
    for name in attributes:
        by_layer: dict[int, float] = {}
        for row in rows_by_attribute[name]:
            score = row.get(metric)
            if score is None:
                continue
            score = float(score)
            if subtract_baseline:
                baseline = row.get(f"baseline_{metric}")
                if baseline is not None:
                    score = score - float(baseline)
            if metric == "mae":  # lower mae is better; flip for the heatmap
                score = -score
            layer = int(row["layer"])
            current = by_layer.get(layer)
            if current is None or score > current:
                by_layer[layer] = score
        per_attribute[name] = by_layer
    layers = sorted({layer for scores in per_attribute.values() for layer in scores})
    if not layers:
        raise ValueError(f"No rows with metric {metric!r} found in any attribute")

    matrix = np.full((len(attributes), len(layers)), np.nan, dtype=float)
    for i, name in enumerate(attributes):
        for j, layer in enumerate(layers):
            value = per_attribute[name].get(layer)
            if value is not None:
                matrix[i, j] = value

    label = (f"{metric} − baseline" if subtract_baseline else metric) + (
        " (flipped; higher is better)" if metric == "mae" else ""
    )
    auto_title = title or (
        f"Attribute × layer selectivity ({label})"
        if subtract_baseline
        else f"Attribute × layer {metric}"
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=layers,
            y=attributes,
            colorscale=colorscale,
            zmid=zmid if subtract_baseline else None,
            colorbar=dict(title=label),
            hovertemplate=(
                "Attribute: %{y}<br>Layer: %{x}<br>"
                f"{label}: "
                "%{z:.3f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        xaxis_title="Layer",
        yaxis_title="Attribute",
        template="plotly_white",
    )
    return apply_fig_fonts(fig, title=auto_title)
