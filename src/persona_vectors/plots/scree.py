"""PCA scree plot."""

import numpy as np
import plotly.graph_objects as go

from persona_vectors.plots._common import finalize, label_color_map


def plot_scree(
    variance_by_condition: dict[str, np.ndarray],
    title: str = "PCA Explained Variance",
    n_components: int = 20,
    cumulative: bool = True,
    filename: str | None = None,
    show: bool = False,
) -> go.Figure:
    """Plot PCA explained variance ratios for one or more conditions."""
    fig = go.Figure()
    colors = label_color_map(list(variance_by_condition))
    for label, variance in variance_by_condition.items():
        values = variance[:n_components]
        components = list(range(1, len(values) + 1))
        color = colors[label]
        fig.add_trace(
            go.Scatter(
                x=components,
                y=values.tolist(),
                mode="lines+markers",
                name=label,
                marker=dict(size=5),
                line=dict(color=color),
            )
        )
        if cumulative:
            fig.add_trace(
                go.Scatter(
                    x=components,
                    y=np.cumsum(values).tolist(),
                    mode="lines",
                    name=f"{label} cumulative",
                    line=dict(color=color, dash="dash"),
                )
            )

    fig.update_layout(
        title=title,
        xaxis_title="Principal component",
        yaxis_title="Explained variance ratio",
        template="plotly_white",
        hovermode="x",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02),
    )

    finalize(fig, filename, show)
    return fig
