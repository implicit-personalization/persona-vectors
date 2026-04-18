from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from persona_data.environment import get_artifacts_dir
from plotly.subplots import make_subplots


def _plots_dir() -> Path:
    path = get_artifacts_dir() / "plots"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_plot_html(fig: go.Figure, filename: str) -> Path:
    """Save a Plotly figure as an HTML artifact."""

    output_path = _plots_dir() / f"{filename}.html"
    fig.write_html(str(output_path))
    return output_path


def save_plot_png(fig: go.Figure, filename: str) -> Path:
    """Save a Plotly figure as a PNG artifact."""

    output_path = _plots_dir() / f"{filename}.png"
    fig.write_image(str(output_path))
    return output_path


def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    return np.asarray(x)


def _similarity_coloraxis() -> dict:
    return dict(
        cmin=-1,
        cmax=1,
        colorscale="RdBu",
        colorbar=dict(title="Cosine sim"),
    )


def _finalize(fig: go.Figure, filename: str | None, show: bool) -> None:
    if filename is not None:
        output_path = save_plot_html(fig, filename)
        print(f"Plot saved to {output_path}")
    if show:
        fig.show()


def _similarity_heatmap(
    z: np.ndarray,
    labels: list[str],
) -> go.Heatmap:
    return go.Heatmap(
        z=z,
        x=labels,
        y=labels,
        coloraxis="coloraxis",
        texttemplate="%{z:.2f}",
        textfont=dict(color="#111111", size=10),
        hovertemplate="(%{x}, %{y})<br>Cosine sim: %{z:.4f}<extra></extra>",
    )


def _add_similarity_traces(
    fig: go.Figure,
    short: torch.Tensor,
    long: torch.Tensor,
    label: str | None = None,
) -> None:
    """Add one cosine-similarity trace to an existing figure."""
    similarities = F.cosine_similarity(short, long, dim=1).tolist()
    layers = list(range(len(similarities)))
    fig.add_trace(
        go.Scatter(
            x=layers,
            y=similarities,
            mode="lines+markers",
            marker=dict(size=5),
            name=label,
            hovertemplate="Layer %{x}<br>Cosine sim: %{y:.4f}<extra></extra>",
        )
    )


def plot_layer_similarity(
    traces: list[tuple[str, torch.Tensor, torch.Tensor]],
    title: str = "Layer-wise Activation Similarity",
    filename: str | None = None,
    show: bool = False,
) -> go.Figure:
    """Plot cosine similarity across layers for one or more (label, short, long) pairs.

    Args:
        traces: List of (label, short, long) tuples. Each label is used for the
            legend entry; short and long are (L, hidden_size) tensors.
            Pass a single-element list for a single-trace plot.
        title: Plot title.
        filename: If provided, save an interactive HTML file as
            <artifacts_dir>/plots/<filename>.html.
        show: If True, open the plot in the browser.

    Returns:
        The Plotly figure object.
    """
    fig = go.Figure()
    for label, short, long in traces:
        _add_similarity_traces(fig, short, long, label=label)
    fig.update_layout(
        title=title,
        xaxis_title="Layer",
        yaxis_title="Cosine similarity",
        hovermode="x",
        template="plotly_white",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
        ),
    )

    _finalize(fig, filename, show)

    return fig


def plot_similarity_matrix(
    sim_matrix: torch.Tensor | np.ndarray,
    labels: list[str],
    title: str = "Pairwise Cosine Similarity",
    filename: str | None = None,
    show: bool = False,
) -> go.Figure:
    """Plot a pairwise cosine similarity matrix as a heatmap."""

    z = _to_numpy(sim_matrix)
    fig = go.Figure(data=_similarity_heatmap(z, labels))
    n_labels = len(labels)
    fig.update_layout(
        title=title,
        template="plotly_white",
        coloraxis=_similarity_coloraxis(),
        width=max(400, 60 * n_labels),
        height=max(400, 60 * n_labels),
        margin=dict(t=100, l=70, r=40, b=70),
    )
    fig.update_xaxes(
        side="top",
        tickangle=-45,
        automargin=True,
        showticklabels=True,
        tickfont=dict(size=9),
    )
    fig.update_yaxes(
        side="left",
        autorange="reversed",
        automargin=True,
        showticklabels=True,
        tickfont=dict(size=9),
    )

    _finalize(fig, filename, show)
    return fig


def plot_similarity_matrix_grid(
    matrices: list[torch.Tensor | np.ndarray],
    labels: list[str],
    titles: list[str],
    title: str = "Pairwise Cosine Similarity",
    filename: str | None = None,
    show: bool = False,
) -> go.Figure:
    """Plot multiple similarity matrices in a 2x2 grid."""

    if len(matrices) != 4 or len(titles) != 4:
        raise ValueError("matrices and titles must both have length 4")

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=titles,
        horizontal_spacing=0.07,
        vertical_spacing=0.09,
    )
    for index, matrix in enumerate(matrices):
        row = index // 2 + 1
        col = index % 2 + 1
        z = _to_numpy(matrix)
        fig.add_trace(_similarity_heatmap(z, labels), row=row, col=col)

    n_labels = len(labels)
    fig.update_layout(
        title=title,
        template="plotly_white",
        coloraxis=_similarity_coloraxis(),
        width=max(600, 90 * n_labels),
        height=max(700, 100 * n_labels),
        margin=dict(t=250, l=70, r=50, b=70),
    )
    for row in [1, 2]:
        for col in [1, 2]:
            fig.update_xaxes(
                side="top",
                tickangle=-45,
                automargin=True,
                showticklabels=row == 1,
                tickfont=dict(size=9),
                col=col,
                row=row,
            )
            fig.update_yaxes(
                side="left",
                autorange="reversed",
                automargin=True,
                showticklabels=col == 1,
                tickfont=dict(size=9),
                col=col,
                row=row,
            )

    _finalize(fig, filename, show)
    return fig
