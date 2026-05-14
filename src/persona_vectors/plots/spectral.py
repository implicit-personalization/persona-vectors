"""Laplacian eigengap diagnostic for spectral clustering."""

import numpy as np
import plotly.graph_objects as go

from persona_vectors.plots._common import finalize, label_color_map


def plot_laplacian_eigengap(
    eigenvalues_by_condition: dict[str, np.ndarray],
    title: str = "Laplacian Eigengap",
    n_components: int = 20,
    filename: str | None = None,
    show: bool = False,
) -> go.Figure:
    """Plot sorted Laplacian eigenvalues; the largest gap suggests the cluster count."""
    fig = go.Figure()
    colors = label_color_map(list(eigenvalues_by_condition))
    for label, eigenvalues in eigenvalues_by_condition.items():
        values = np.sort(np.asarray(eigenvalues))[:n_components]
        indices = list(range(1, len(values) + 1))
        color = colors[label]
        fig.add_trace(
            go.Scatter(
                x=indices,
                y=values.tolist(),
                mode="lines+markers",
                name=label,
                marker=dict(size=5),
                line=dict(color=color),
            )
        )
        if len(values) >= 2:
            suggested_k = int(np.argmax(np.diff(values)) + 1)
            fig.add_trace(
                go.Scatter(
                    x=[suggested_k],
                    y=[float(values[suggested_k - 1])],
                    mode="markers",
                    name=f"{label} suggested k={suggested_k}",
                    marker=dict(size=12, symbol="circle-open", color=color),
                )
            )

    fig.update_layout(
        title=title,
        xaxis_title="Eigenvalue index",
        yaxis_title="Eigenvalue",
        template="plotly_white",
        hovermode="x",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02),
    )

    finalize(fig, filename, show)
    return fig
