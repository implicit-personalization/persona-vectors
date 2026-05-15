"""Cosine-similarity plots: per-layer line, layered heatmap, pair trajectory."""

import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn.functional as F

from persona_vectors.analysis import LayeredSamples, cosine_similarity_matrix
from persona_vectors.plots._common import (
    apply_fig_fonts,
    finalize,
    layer_animation_buttons,
    layer_frame_layout,
    layer_slider,
    validate_layers,
)


def _similarity_heatmap(
    z: np.ndarray,
    labels: list[str],
    hover_label: str = "Cosine sim",
) -> go.Heatmap:
    return go.Heatmap(
        z=z,
        x=labels,
        y=labels,
        coloraxis="coloraxis",
        texttemplate="%{z:.2f}",
        textfont=dict(color="#111827", size=10),
        hovertemplate=f"(%{{x}}, %{{y}})<br>{hover_label}: %{{z:.4f}}<extra></extra>",
    )


def layer_cosine_matrices(
    vectors: torch.Tensor, layers: list[int]
) -> dict[int, np.ndarray]:
    return {
        layer: cosine_similarity_matrix(vectors[:, layer, :], center=True).cpu().numpy()
        for layer in layers
    }


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
        similarities = F.cosine_similarity(short, long, dim=1).tolist()
        fig.add_trace(
            go.Scatter(
                x=list(range(len(similarities))),
                y=similarities,
                mode="lines+markers",
                marker=dict(size=5),
                name=label,
                hovertemplate="Layer %{x}<br>Cosine sim: %{y:.4f}<extra></extra>",
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Layer",
        yaxis_title="Cosine similarity",
        hovermode="x",
        template="plotly_white",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02),
    )

    apply_fig_fonts(fig)
    finalize(fig, filename, show)
    return fig


def build_layered_similarity_figure(
    samples: LayeredSamples,
    selected_layers: list[int],
    title: str = "Centered Cosine Similarity by Layer",
    matrices: dict[int, np.ndarray] | None = None,
) -> go.Figure:
    if matrices is None:
        matrices = layer_cosine_matrices(samples.vectors, selected_layers)
    first_layer = selected_layers[0]
    fig = go.Figure(
        data=[
            _similarity_heatmap(
                matrices[first_layer], samples.labels, hover_label="Centered cosine"
            )
        ]
    )
    frames = [
        go.Frame(
            name=str(layer),
            data=[
                _similarity_heatmap(
                    matrices[layer], samples.labels, hover_label="Centered cosine"
                )
            ],
            layout=layer_frame_layout(title, layer),
        )
        for layer in selected_layers
    ]
    fig.frames = frames
    fig.update_layout(
        title={
            "text": f"{title} - Layer {first_layer}",
            "font": {"size": 24},
            "y": 0.98,
            "yanchor": "top",
        },
        template="plotly_white",
        width=max(800, 26 * len(samples.labels)),
        height=max(700, 24 * len(samples.labels)),
        margin=dict(t=170, b=90),
        coloraxis=dict(
            cmin=-1,
            cmax=1,
            cmid=0,
            colorscale="RdBu_r",
            colorbar=dict(title="Cosine sim"),
        ),
        updatemenus=layer_animation_buttons(),
        sliders=layer_slider(selected_layers),
    )
    fig.update_xaxes(side="top", tickangle=-45, automargin=True)
    fig.update_yaxes(autorange="reversed", automargin=True)
    apply_fig_fonts(fig)
    return fig


def _build_pair_similarity_figure(
    samples: LayeredSamples,
    selected_layers: list[int],
    matrices: dict[int, np.ndarray],
    title: str = "Centered Pair Similarity Across Layers",
) -> go.Figure:
    n_samples = samples.vectors.shape[0]
    if n_samples < 2:
        raise ValueError("At least two samples are required")

    pairs = [
        (left, right)
        for left in range(n_samples)
        for right in range(left + 1, n_samples)
    ]
    show_legend = len(pairs) <= 30

    fig = go.Figure()
    for left, right in pairs:
        left_label = samples.labels[left]
        right_label = samples.labels[right]
        pair_label = f"{left_label} <> {right_label}"
        values = [float(matrices[layer][left, right]) for layer in selected_layers]
        fig.add_trace(
            go.Scatter(
                x=selected_layers,
                y=values,
                mode="lines+markers",
                name=pair_label,
                showlegend=show_legend,
                marker=dict(size=5),
                line=dict(width=1.8),
                opacity=0.78,
                hovertemplate=(
                    f"{left_label}<br>{right_label}<br>"
                    "Layer %{x}<br>Centered cosine: %{y:.4f}<extra></extra>"
                ),
            )
        )

    fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="#64748b")
    fig.update_layout(
        title=title,
        xaxis_title="Layer",
        yaxis_title="Centered cosine similarity",
        yaxis=dict(range=[-1, 1]),
        hovermode="closest",
        template="plotly_white",
        margin=dict(t=90, b=70),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02),
    )
    fig.update_xaxes(tickmode="array", tickvals=selected_layers, automargin=True)
    fig.update_yaxes(zeroline=True, automargin=True)
    apply_fig_fonts(fig)
    return fig


def build_pair_similarity_figure(
    samples: LayeredSamples,
    layers: list[int] | None = None,
    title: str = "Centered Pair Similarity Across Layers",
) -> go.Figure:
    """Plot each persona-pair similarity as a line across layers."""
    selected_layers = validate_layers(samples.vectors, layers)
    matrices = layer_cosine_matrices(samples.vectors, selected_layers)
    return _build_pair_similarity_figure(samples, selected_layers, matrices, title)


def build_similarity_figures(
    samples: LayeredSamples,
    layers: list[int] | None = None,
    title: str = "Centered Cosine Similarity by Layer",
    pair_title: str = "Centered Pair Similarity Across Layers",
) -> tuple[go.Figure, go.Figure]:
    """Build similarity heatmap and pair trajectory figures from one matrix pass."""
    selected_layers = validate_layers(samples.vectors, layers)
    matrices = layer_cosine_matrices(samples.vectors, selected_layers)
    return (
        build_layered_similarity_figure(samples, selected_layers, title, matrices),
        _build_pair_similarity_figure(samples, selected_layers, matrices, pair_title),
    )
