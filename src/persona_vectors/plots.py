from pathlib import Path

import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from persona_data.environment import get_artifacts_dir


def _plots_dir():
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
    short: torch.Tensor,
    long: torch.Tensor,
    title: str = "Layer-wise Activation Similarity",
    filename: str | None = None,
    show: bool = False,
) -> go.Figure:
    """Plot cosine similarity between two sets of activation across layers.

    Args:
        short: (L, hidden_size) tensor of activations for the first prompt.
        long: (L, hidden_size) tensor of activations for the second prompt.
        title: Plot title.
        filename: If provided, save an interactive HTML file as
            <artifacts_dir>/plots/<filename>.html.
        show: If True, open the plot in the browser.

    Returns:
        The Plotly figure object.
    """
    fig = go.Figure()
    _add_similarity_traces(fig, short, long)
    fig.update_layout(
        title=title,
        xaxis_title="Layer",
        yaxis_title="Cosine similarity",
        hovermode="x",
        template="plotly_white",
    )

    if filename is not None:
        output_path = save_plot_html(fig, filename)
        print(f"Plot saved to {output_path}")

    if show:
        fig.show()

    return fig


def plot_multiple_layer_similarities(
    traces: list[tuple[str, torch.Tensor, torch.Tensor]],
    title: str = "Layer-wise Activation Similarity",
    filename: str | None = None,
    show: bool = False,
) -> go.Figure:
    """Plot cosine similarity across layers for multiple personas on one figure.

    Args:
        traces: List of (label, short, long) tuples. Each label is used for the
            legend entry; short and long are (L, hidden_size) tensors as in
            plot_layer_similarity.
        title: Plot title.
        filename: If provided, save an interactive HTML file.
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

    if filename is not None:
        output_path = save_plot_html(fig, filename)
        print(f"Plot saved to {output_path}")

    if show:
        fig.show()

    return fig
