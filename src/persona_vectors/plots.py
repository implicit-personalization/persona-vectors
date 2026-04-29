from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from persona_data.environment import get_artifacts_dir
from plotly.colors import qualitative
from plotly.subplots import make_subplots

from persona_vectors.analysis import (
    LayeredSamples,
    cosine_similarity_matrix,
    project_pca_centered,
    project_umap_centered,
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


def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    return np.asarray(x)


def _similarity_coloraxis() -> dict:
    return dict(
        cmin=-1,
        cmax=1,
        cmid=0,
        colorscale="RdBu_r",
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


def _centered_similarity_heatmap(z: np.ndarray, labels: list[str]) -> go.Heatmap:
    return _similarity_heatmap(z, labels, hover_label="Centered cosine")


def _label_color_map(labels: list[str]) -> dict[str, str]:
    palette = qualitative.Safe + qualitative.Dark24 + qualitative.Set3
    unique_labels = sorted(set(labels), key=lambda value: value.casefold())
    return {
        label: palette[index % len(palette)]
        for index, label in enumerate(unique_labels)
    }


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
    fig.update_layout(
        title=title,
        template="plotly_white",
        coloraxis=_similarity_coloraxis(),
    )
    fig.update_xaxes(
        side="top",
        tickangle=-45,
        automargin=True,
        showticklabels=True,
    )
    fig.update_yaxes(
        side="left",
        autorange="reversed",
        automargin=True,
        showticklabels=True,
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


# ---------------------------------------------------------------------------
# Scree / elbow plot for PCA explained variance
# ---------------------------------------------------------------------------


def plot_scree(
    variance_by_condition: dict[str, np.ndarray],
    title: str = "PCA Explained Variance (Scree Plot)",
    n_components: int = 20,
    cumulative: bool = True,
    filename: str | None = None,
    show: bool = False,
) -> go.Figure:
    """Plot explained variance ratio per principal component for multiple conditions.

    Args:
        variance_by_condition: Mapping from condition label (e.g. "baseline",
            "templated", "biography") to an array of explained variance ratios.
        title: Plot title.
        n_components: How many components to display.
        cumulative: If True, also plot cumulative variance.
        filename: If provided, save as HTML.
        show: If True, open in browser.
    """
    fig = go.Figure()
    colors = _label_color_map(list(variance_by_condition))
    for label, var in variance_by_condition.items():
        var = var[:n_components]
        components = list(range(1, len(var) + 1))
        color = colors[label]
        fig.add_trace(
            go.Scatter(
                x=components,
                y=var.tolist(),
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
                    y=np.cumsum(var).tolist(),
                    mode="lines",
                    name=f"{label} (cumulative)",
                    line=dict(color=color, dash="dash"),
                )
            )

    fig.update_layout(
        title=title,
        xaxis_title="Principal Component",
        yaxis_title="Explained Variance Ratio",
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
        data=[_centered_similarity_heatmap(matrices[first_layer], samples.labels)]
    )
    frames = []
    for layer in selected_layers:
        frames.append(
            go.Frame(
                name=str(layer),
                data=[_centered_similarity_heatmap(matrices[layer], samples.labels)],
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
        coloraxis=_similarity_coloraxis(),
        updatemenus=_layer_animation_buttons(),
        sliders=_layer_slider(selected_layers),
    )
    fig.update_xaxes(side="top", tickangle=-45, automargin=True)
    fig.update_yaxes(autorange="reversed", automargin=True)
    return fig


def build_layered_figure(
    samples: LayeredSamples,
    kind: str,
    layers: list[int] | None = None,
    title: str | None = None,
) -> go.Figure:
    """Build an interactive per-layer PCA, UMAP, or similarity figure."""

    selected_layers = _validate_layers(samples.vectors, layers)
    if samples.vectors.shape[0] < 2:
        raise ValueError("At least two samples are required")
    if kind == "pca":
        return _build_layered_embedding_figure(
            samples,
            selected_layers,
            title=title or "Centered PCA by Layer",
            project_fn=project_pca_centered,
            x_label="PC1",
            y_label="PC2",
        )
    if kind == "umap":
        return _build_layered_embedding_figure(
            samples,
            selected_layers,
            title=title or "Centered UMAP by Layer",
            project_fn=project_umap_centered,
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


def build_embedding_figure(
    coords: torch.Tensor,
    labels: list[str],
    title: str,
    x_label: str,
    y_label: str,
    hover_text: list[str] | None = None,
) -> go.Figure:
    """Build a 2D scatter plot from projected coordinates."""
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must have shape (n_samples, 2)")
    if len(labels) != coords.shape[0]:
        raise ValueError("labels must match number of samples")
    if hover_text is not None and len(hover_text) != coords.shape[0]:
        raise ValueError("hover_text must match number of samples")

    fig = go.Figure()
    unique_labels = sorted(set(labels), key=lambda value: value.casefold())
    label_colors = _label_color_map(labels)

    for label in unique_labels:
        mask = torch.tensor([value == label for value in labels], dtype=torch.bool)
        selected = coords[mask]
        fig.add_trace(
            go.Scatter(
                x=selected[:, 0].tolist(),
                y=selected[:, 1].tolist(),
                mode="markers",
                name=label,
                marker=dict(
                    size=8,
                    opacity=0.8,
                    color=label_colors[label],
                ),
                text=(
                    [hover_text[i] for i, value in enumerate(labels) if value == label]
                    if hover_text is not None
                    else None
                ),
                hovertemplate="%{text}<br>x=%{x:.4f}<br>y=%{y:.4f}<extra></extra>",
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        template="plotly_white",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02),
    )
    return fig
