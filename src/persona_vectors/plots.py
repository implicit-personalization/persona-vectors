from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from persona_data.environment import get_artifacts_dir
from plotly.colors import qualitative

from persona_vectors.analysis import (
    LayeredSamples,
    cosine_similarity_matrix,
    project_pca,
    project_umap,
)


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


def _finalize(fig: go.Figure, filename: str | None, show: bool) -> None:
    if filename is not None:
        output_path = save_plot_html(fig, filename)
        print(f"Plot saved to {output_path}")
    if show:
        fig.show()


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


def _label_color_map(labels: list[str]) -> dict[str, str]:
    palette = qualitative.Safe + qualitative.Dark24 + qualitative.Set3
    unique_labels = sorted(set(labels), key=lambda value: value.casefold())
    return {
        label: palette[index % len(palette)]
        for index, label in enumerate(unique_labels)
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

    _finalize(fig, filename, show)
    return fig


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
    colors = _label_color_map(list(variance_by_condition))
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

    _finalize(fig, filename, show)
    return fig


def _validate_layers(vectors: torch.Tensor, layers: list[int] | None) -> list[int]:
    if vectors.ndim != 3:
        raise ValueError("vectors must have shape (n_samples, num_layers, hidden_size)")
    num_layers = int(vectors.shape[1])
    selected = list(range(num_layers)) if layers is None else list(layers)
    invalid = [layer for layer in selected if layer < 0 or layer >= num_layers]
    if invalid:
        raise ValueError(
            f"Invalid layer(s) for tensor with {num_layers} layers: {invalid}"
        )
    return selected


def _layer_slider(selected_layers: list[int]) -> list[dict]:
    return [
        dict(
            active=0,
            currentvalue=dict(prefix="Layer "),
            pad=dict(t=45),
            steps=[
                dict(
                    label=str(layer),
                    method="animate",
                    args=[
                        [str(layer)],
                        dict(
                            mode="immediate",
                            frame=dict(duration=0, redraw=True),
                            transition=dict(duration=0),
                        ),
                    ],
                )
                for layer in selected_layers
            ],
        )
    ]


def _layer_animation_buttons() -> list[dict]:
    return [
        dict(
            type="buttons",
            direction="left",
            active=-1,
            x=0,
            xanchor="left",
            y=1.18,
            yanchor="top",
            bgcolor="#f8fafc",
            bordercolor="#94a3b8",
            font=dict(color="#111827", size=13),
            pad=dict(t=0, r=10),
            buttons=[
                dict(
                    label="Play",
                    method="animate",
                    args=[
                        None,
                        dict(
                            frame=dict(duration=650, redraw=True),
                            transition=dict(duration=250),
                            fromcurrent=True,
                        ),
                    ],
                ),
                dict(
                    label="Pause",
                    method="animate",
                    args=[
                        [None],
                        dict(
                            mode="immediate",
                            frame=dict(duration=0, redraw=False),
                            transition=dict(duration=0),
                        ),
                    ],
                ),
            ],
        )
    ]


def _layer_frame_layout(
    title: str,
    layer: int,
    x_range: list[float] | None = None,
    y_range: list[float] | None = None,
) -> dict:
    layout = {
        "title": {
            "text": f"{title} - Layer {layer}",
            "font": {"size": 24},
            "y": 0.98,
            "yanchor": "top",
        }
    }
    if x_range is not None:
        layout["xaxis"] = {"range": x_range}
    if y_range is not None:
        layout["yaxis"] = {"range": y_range}
    return layout


def _coordinate_range(coords: torch.Tensor, axis: int) -> list[float]:
    values = coords[:, axis].float().cpu()
    low = float(values.min())
    high = float(values.max())
    if low == high:
        padding = 1.0
    else:
        padding = (high - low) * 0.08
    return [low - padding, high + padding]


def _build_layered_embedding_figure(
    samples: LayeredSamples,
    selected_layers: list[int],
    title: str,
    project_fn,
    x_label: str,
    y_label: str,
) -> go.Figure:
    layer_coords = {
        layer: project_fn(samples.vectors[:, layer, :]) for layer in selected_layers
    }
    layer_ranges = {
        layer: (
            _coordinate_range(coords, 0),
            _coordinate_range(coords, 1),
        )
        for layer, coords in layer_coords.items()
    }
    unique_labels = sorted(set(samples.labels), key=lambda value: value.casefold())
    label_colors = _label_color_map(samples.labels)
    label_indices = {
        label: [index for index, value in enumerate(samples.labels) if value == label]
        for label in unique_labels
    }

    first_layer = selected_layers[0]
    first_coords = layer_coords[first_layer]
    traces = []
    for label in unique_labels:
        indices = label_indices[label]
        traces.append(
            go.Scatter(
                x=first_coords[indices, 0].tolist(),
                y=first_coords[indices, 1].tolist(),
                mode="markers",
                name=label,
                marker=dict(
                    size=9,
                    opacity=0.82,
                    color=label_colors[label],
                ),
                text=[samples.hover_text[index] for index in indices],
                hovertemplate=(
                    "%{text}<br>Group: " + label + "<br>"
                    f"{x_label}=%{{x:.4f}}<br>{y_label}=%{{y:.4f}}<extra></extra>"
                ),
            )
        )
    frames = []
    for layer in selected_layers:
        layer_xy = layer_coords[layer]
        x_range, y_range = layer_ranges[layer]
        frames.append(
            go.Frame(
                name=str(layer),
                data=[
                    go.Scatter(
                        x=layer_xy[label_indices[label], 0].tolist(),
                        y=layer_xy[label_indices[label], 1].tolist(),
                    )
                    for label in unique_labels
                ],
                layout=_layer_frame_layout(title, layer, x_range, y_range),
            )
        )

    fig = go.Figure(data=traces, frames=frames)
    fig.update_layout(
        title={
            "text": f"{title} - Layer {first_layer}",
            "font": {"size": 24},
            "y": 0.98,
            "yanchor": "top",
        },
        xaxis_title=x_label,
        yaxis_title=y_label,
        template="plotly_white",
        margin=dict(t=140, b=90),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02),
        updatemenus=_layer_animation_buttons(),
        sliders=_layer_slider(selected_layers),
    )
    first_x_range, first_y_range = layer_ranges[first_layer]
    fig.update_xaxes(range=first_x_range, zeroline=True, automargin=True)
    fig.update_yaxes(range=first_y_range, zeroline=True, automargin=True)
    return fig


def _build_layered_similarity_figure(
    samples: LayeredSamples,
    selected_layers: list[int],
    title: str = "Centered Cosine Similarity by Layer",
) -> go.Figure:
    matrices = {
        layer: cosine_similarity_matrix(samples.vectors[:, layer, :], center=True)
        .cpu()
        .numpy()
        for layer in selected_layers
    }
    first_layer = selected_layers[0]
    fig = go.Figure(
        data=[
            _similarity_heatmap(
                matrices[first_layer], samples.labels, hover_label="Centered cosine"
            )
        ]
    )
    frames = []
    for layer in selected_layers:
        frames.append(
            go.Frame(
                name=str(layer),
                data=[
                    _similarity_heatmap(
                        matrices[layer], samples.labels, hover_label="Centered cosine"
                    )
                ],
                layout=_layer_frame_layout(title, layer),
            )
        )
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
        updatemenus=_layer_animation_buttons(),
        sliders=_layer_slider(selected_layers),
    )
    fig.update_xaxes(side="top", tickangle=-45, automargin=True)
    fig.update_yaxes(autorange="reversed", automargin=True)
    return fig


def build_pair_similarity_figure(
    samples: LayeredSamples,
    layers: list[int] | None = None,
    title: str = "Centered Pair Similarity Across Layers",
) -> go.Figure:
    """Plot each persona-pair similarity as a line across layers."""

    selected_layers = _validate_layers(samples.vectors, layers)
    n_samples = samples.vectors.shape[0]
    if n_samples < 2:
        raise ValueError("At least two samples are required")

    matrices = {
        layer: cosine_similarity_matrix(samples.vectors[:, layer, :], center=True)
        .cpu()
        .numpy()
        for layer in selected_layers
    }
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
    return fig


def build_layered_figure(
    samples: LayeredSamples,
    kind: str,
    layers: list[int] | None = None,
    title: str | None = None,
) -> go.Figure:
    """Build an interactive per-layer PCA, UMAP, or similarity figure.

    This is the main plotting entry point for persona-space views. It accepts
    the ``LayeredSamples`` returned by analysis helpers and adds the shared
    layer slider/animation controls used by all layered plots.
    """

    selected_layers = _validate_layers(samples.vectors, layers)
    if samples.vectors.shape[0] < 2:
        raise ValueError("At least two samples are required")
    if kind == "pca":
        return _build_layered_embedding_figure(
            samples,
            selected_layers,
            title=title or "PCA by Layer",
            project_fn=project_pca,
            x_label="PC1",
            y_label="PC2",
        )
    if kind == "umap":
        return _build_layered_embedding_figure(
            samples,
            selected_layers,
            title=title or "Centered UMAP by Layer",
            project_fn=project_umap,
            x_label="UMAP 1",
            y_label="UMAP 2",
        )
    if kind == "similarity":
        return _build_layered_similarity_figure(
            samples,
            selected_layers,
            title=title or "Centered Cosine Similarity by Layer",
        )
    raise ValueError("kind must be one of: pca, umap, similarity")
