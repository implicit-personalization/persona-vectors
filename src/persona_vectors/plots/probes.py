"""Plots for probe sweeps.

Kept separate from the projection/similarity machinery in ``plots`` so the
probe notebooks (and the persona-ui Probing tab) can import a small,
self-contained surface.
"""

import plotly.graph_objects as go

__all__ = ["plot_metric_over_layers"]


def plot_metric_over_layers(
    rows: list[dict[str, object]],
    attribute_name: str,
    *,
    metric: str = "balanced_accuracy",
) -> go.Figure:
    """Line plot of one metric over layers, one trace per (probe_kind, space).

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

    keys = sorted(
        {
            (row["probe_kind"], row["feature_space"])
            for row in attr_rows
            if row.get(metric) is not None
        }
    )
    for probe_kind, feature_space in keys:
        series = [
            row
            for row in attr_rows
            if row["probe_kind"] == probe_kind
            and row["feature_space"] == feature_space
            and row.get(metric) is not None
        ]
        if not series:
            continue
        fig.add_trace(
            go.Scatter(
                x=[row["layer"] for row in series],
                y=[row[metric] for row in series],
                mode="lines+markers",
                name=f"{probe_kind} / {feature_space}",
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
