"""Plots for probe sweeps.

Kept separate from the projection/similarity machinery in ``plots`` so the
probe notebooks (and the persona-ui Probing tab) can import a small,
self-contained surface.
"""

from collections.abc import Iterable, Mapping

import numpy as np
import plotly.graph_objects as go

__all__ = [
    "plot_attribute_layer_selectivity_heatmap",
    "plot_metric_comparison",
    "plot_metric_over_layers",
]


def plot_metric_over_layers(
    rows: list[dict[str, object]],
    attribute_name: str,
    *,
    metric: str = "balanced_accuracy",
) -> go.Figure:
    """Line plot of one metric over layers, one trace per probe_kind.

    The majority-class / mean-prediction baseline, when present, is drawn as a
    dotted horizontal line. Metrics in [0, 1] get a fixed y-range; ``mae`` and
    ``r2`` are autoscaled.
    """
    fig = go.Figure()
    attr_rows = [row for row in rows if row["attribute"] == attribute_name]
    if not attr_rows:
        return fig

    baseline = attr_rows[0].get(f"baseline_{metric}")
    if baseline is not None:
        fig.add_hline(
            y=baseline,
            line=dict(color="#64748b", dash="dot", width=2),
            annotation_text=f"baseline {baseline:.3f}",
            annotation_position="top left",
        )

    probe_kinds = sorted(
        {row["probe_kind"] for row in attr_rows if row.get(metric) is not None}
    )
    for probe_kind in probe_kinds:
        series = [
            row
            for row in attr_rows
            if row["probe_kind"] == probe_kind and row.get(metric) is not None
        ]
        if not series:
            continue
        fig.add_trace(
            go.Scatter(
                x=[row["layer"] for row in series],
                y=[row[metric] for row in series],
                mode="lines+markers",
                name=str(probe_kind),
                hovertemplate=f"Layer %{{x}}<br>{metric}: %{{y:.3f}}<extra></extra>",
            )
        )

    y_range = [0.0, 1.02] if metric not in {"r2", "mae"} else None
    fig.update_layout(
        title=f"{attribute_name}: {metric} over layers",
        xaxis_title="Layer",
        yaxis_title=metric,
        yaxis=dict(range=y_range),
        template="plotly_white",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02),
    )
    return fig


_COMPARISON_DASHES = ("solid", "dash", "dot", "dashdot")


def plot_metric_comparison(
    rows_by_label: Mapping[str, list[dict[str, object]]],
    attribute_name: str,
    *,
    metric: str = "balanced_accuracy",
) -> go.Figure:
    """Overlay one metric over layers for several labelled sweeps.

    ``rows_by_label`` maps a short label (e.g. ``"full"``, ``"pca10"``) to a
    sweep's rows. One trace per (label, probe_kind); each label gets its own
    line style so full vs compressed features are easy to compare in a single
    figure.
    """
    fig = go.Figure()
    baseline_drawn = False
    for dash, (label, rows) in zip(
        _COMPARISON_DASHES, rows_by_label.items(), strict=False
    ):
        attr_rows = [
            row
            for row in rows
            if row["attribute"] == attribute_name and row.get(metric) is not None
        ]
        if not attr_rows:
            continue
        if not baseline_drawn:
            baseline = attr_rows[0].get(f"baseline_{metric}")
            if baseline is not None:
                fig.add_hline(
                    y=baseline,
                    line=dict(color="#64748b", dash="dot", width=2),
                    annotation_text=f"baseline {baseline:.3f}",
                    annotation_position="top left",
                )
            baseline_drawn = True
        for probe_kind in sorted({row["probe_kind"] for row in attr_rows}):
            series = [r for r in attr_rows if r["probe_kind"] == probe_kind]
            fig.add_trace(
                go.Scatter(
                    x=[r["layer"] for r in series],
                    y=[r[metric] for r in series],
                    mode="lines+markers",
                    name=f"{label} · {probe_kind}",
                    line=dict(dash=dash),
                    hovertemplate=(
                        f"{label} / {probe_kind}<br>Layer %{{x}}<br>"
                        f"{metric}: %{{y:.3f}}<extra></extra>"
                    ),
                )
            )

    y_range = [0.0, 1.02] if metric not in {"r2", "mae"} else None
    fig.update_layout(
        title=f"{attribute_name}: {metric} over layers",
        xaxis_title="Layer",
        yaxis_title=metric,
        yaxis=dict(range=y_range),
        template="plotly_white",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02),
    )
    return fig


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

    label = (
        f"{metric} − baseline" if subtract_baseline else metric
    ) + (" (flipped; higher is better)" if metric == "mae" else "")
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
                f"{label}: " "%{z:.3f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=auto_title,
        xaxis_title="Layer",
        yaxis_title="Attribute",
        template="plotly_white",
    )
    return fig
