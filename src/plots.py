from pathlib import Path

import plotly.graph_objects as go
import torch
import torch.nn.functional as F


def plot_layer_similarity(
    short: torch.Tensor,
    long: torch.Tensor,
    title: str = "Layer-wise Activation Similarity",
    output_path: str | None = None,
    show: bool = False,
) -> go.Figure:
    """Plot cosine similarity between two sets of activation across layers.

    Args:
        short: (L, d_model) tensor of activations for the first prompt.
        long: (L, d_model) tensor of activations for the second prompt.
        title: Plot title.
        output_path: If provided, save the interactive HTML file to this path.
        show: If True, open the plot in the browser.

    Returns:
        The Plotly figure object.
    """
    similarities = F.cosine_similarity(short, long, dim=1).tolist()
    layers = list(range(len(similarities)))

    fig = go.Figure(
        go.Scatter(
            x=layers,
            y=similarities,
            mode="lines+markers",
            marker=dict(size=5),
            line=dict(color="blue"),
            hovertemplate="Layer %{x}<br>Cosine sim: %{y:.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Layer",
        yaxis_title="Cosine similarity",
        hovermode="x",
        template="plotly_white",
    )

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(output_path)
        print(f"Plot saved to {output_path}")

    if show:
        fig.show()

    return fig
