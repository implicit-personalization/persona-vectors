from pathlib import Path
from typing import Literal

import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from persona_data.environment import get_artifacts_dir
from plotly.colors import qualitative

from persona_vectors.analysis import (
    LayeredSamples,
    cluster_agglomerative,
    cluster_hdbscan,
    cluster_kmeans,
    cosine_similarity_matrix,
    prepare_cluster_samples,
    prepare_layer_mean_cluster_samples,
    project_pca,
    project_umap,
)

ClusterMode = Literal["mean_across_layers", "first_layer", "per_layer"]
ClusterMethod = Literal["kmeans", "agglomerative", "hdbscan"]


def _plots_dir() -> Path:
    path = get_artifacts_dir() / "plots"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _layer_cosine_matrices(
    vectors: torch.Tensor, layers: list[int]
) -> dict[int, np.ndarray]:
    return {
        layer: cosine_similarity_matrix(vectors[:, layer, :], center=True).cpu().numpy()
        for layer in layers
    }


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
    num_layers = int(vectors.shape[1])
    selected = list(range(num_layers)) if layers is None else list(layers)
    invalid = [layer for layer in selected if layer < 0 or layer >= num_layers]
    if invalid:
        raise ValueError(
            f"Invalid layer(s) for tensor with {num_layers} layers: {invalid}"
        )
    return selected


def _layer_slider(selected_layers: list[int], pad_t: int = 45) -> list[dict]:
    return [
        dict(
            active=0,
            currentvalue=dict(prefix="Layer "),
            pad=dict(t=pad_t),
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
    z_range: list[float] | None = None,
) -> dict:
    layout = {
        "title": {
            "text": f"{title} - Layer {layer}",
            "font": {"size": 24},
            "y": 0.98,
            "yanchor": "top",
        }
    }
    if z_range is not None:
        scene: dict = {}
        if x_range is not None:
            scene["xaxis"] = {"range": x_range}
        if y_range is not None:
            scene["yaxis"] = {"range": y_range}
        scene["zaxis"] = {"range": z_range}
        layout["scene"] = scene
        return layout
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


def _trace_y_range(traces) -> list[float]:
    high = 0.0
    for trace in traces:
        for value in trace.y:
            if value is None:
                continue
            numeric = float(value)
            if np.isfinite(numeric):
                high = max(high, numeric)
    return [0.0, high * 1.08 if high > 0 else 1.0]


def _validate_linkage(linkage: str) -> str:
    if linkage not in {"ward", "average", "complete", "single"}:
        raise ValueError("linkage must be one of: ward, average, complete, single")
    return linkage


def _dendrogram_distance_label(linkage: str, normalize: bool) -> str:
    if linkage == "ward":
        return "Ward distance"
    return "Unit-vector distance" if normalize else "Euclidean distance"


def _dendrogram_title(linkage: str) -> str:
    return f"{linkage.title()} dendrogram"


def _dendrogram_linkage_kwargs(linkage: str) -> dict:
    if linkage == "ward":
        return {"method": "ward"}
    return {"method": linkage, "metric": "euclidean"}


def _create_persona_dendrogram(
    data: torch.Tensor,
    labels: list[str],
    *,
    linkage: str,
    center: bool,
    normalize: bool,
):
    from plotly.figure_factory import create_dendrogram
    from scipy.cluster.hierarchy import linkage as scipy_linkage

    prepared = prepare_cluster_samples(data, center=center, normalize=normalize)
    linkage_kwargs = _dendrogram_linkage_kwargs(linkage)
    return create_dendrogram(
        prepared.cpu().numpy(),
        labels=labels,
        linkagefun=lambda x: scipy_linkage(x, **linkage_kwargs),
    )


def _cluster_label(cluster_id: int) -> str:
    return "Noise" if int(cluster_id) == -1 else f"Cluster {int(cluster_id)}"


def _cluster_projection_samples(
    samples: torch.Tensor,
    *,
    method: ClusterMethod,
    n_clusters: int | None,
    seed: int,
    linkage: str,
    min_cluster_size: int,
    min_samples: int | None,
    center: bool = True,
    normalize: bool = True,
) -> list[str]:
    if method == "kmeans":
        if n_clusters is None:
            raise ValueError("n_clusters is required for kmeans clustering")
        cluster_ids = cluster_kmeans(
            samples,
            n_clusters=n_clusters,
            seed=seed,
            center=center,
            normalize=normalize,
        )
    elif method == "agglomerative":
        if n_clusters is None:
            raise ValueError("n_clusters is required for agglomerative clustering")
        cluster_ids = cluster_agglomerative(
            samples,
            n_clusters=n_clusters,
            linkage=linkage,
            center=center,
            normalize=normalize,
        )
    elif method == "hdbscan":
        cluster_ids = cluster_hdbscan(
            samples,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            center=center,
            normalize=normalize,
        )
    else:
        raise ValueError(
            "cluster_method must be one of: kmeans, agglomerative, hdbscan"
        )
    return [_cluster_label(int(cluster_id)) for cluster_id in cluster_ids]


def _embedding_hovertemplate(
    group_label: str,
    x_label: str,
    y_label: str,
    z_label: str | None,
) -> str:
    template = (
        "%{text}<br>Group: "
        + group_label
        + "<br>"
        + f"{x_label}=%{{x:.4f}}<br>"
        + f"{y_label}=%{{y:.4f}}"
    )
    if z_label is not None:
        template += f"<br>{z_label}=%{{z:.4f}}"
    return template + "<extra></extra>"


def _embedding_trace(
    coords: torch.Tensor,
    indices: list[int],
    *,
    n_components: int,
    name: str | None = None,
    marker: dict | None = None,
    text: list[str] | None = None,
    hovertemplate: str | None = None,
) -> go.Scattergl | go.Scatter3d:
    kwargs = {
        "x": coords[indices, 0].tolist(),
        "y": coords[indices, 1].tolist(),
    }
    if n_components == 3:
        kwargs["z"] = coords[indices, 2].tolist()
        trace_cls = go.Scatter3d
    else:
        trace_cls = go.Scattergl

    if name is not None:
        kwargs["mode"] = "markers"
        kwargs["name"] = name
    if marker is not None:
        kwargs["marker"] = marker
    if text is not None:
        kwargs["text"] = text
    if hovertemplate is not None:
        kwargs["hovertemplate"] = hovertemplate
    return trace_cls(**kwargs)


def _build_layered_projection_figure(
    samples: LayeredSamples,
    selected_layers: list[int],
    title: str,
    project_fn,
    x_label: str,
    y_label: str,
    z_label: str | None = None,
    n_components: int = 2,
    groups: list[str] | dict[int, list[str]] | None = None,
) -> go.Figure:
    if n_components not in (2, 3):
        raise ValueError("n_components must be 2 or 3")

    layer_inputs = [samples.vectors[:, layer, :] for layer in selected_layers]
    coords_list = [
        project_fn(layer_input, n_components=n_components)
        for layer_input in layer_inputs
    ]
    layer_coords = dict(zip(selected_layers, coords_list))
    layer_ranges = {
        layer: tuple(_coordinate_range(coords, axis) for axis in range(n_components))
        for layer, coords in layer_coords.items()
    }

    n_samples = int(samples.vectors.shape[0])
    if groups is None:
        groups_by_layer = {layer: list(samples.labels) for layer in selected_layers}
    elif isinstance(groups, dict):
        missing = [layer for layer in selected_layers if layer not in groups]
        if missing:
            raise ValueError(f"groups is missing layer(s): {missing}")
        groups_by_layer = {layer: list(groups[layer]) for layer in selected_layers}
    else:
        stable_groups = list(groups)
        groups_by_layer = {layer: stable_groups for layer in selected_layers}
    invalid_lengths = {
        layer: len(layer_groups)
        for layer, layer_groups in groups_by_layer.items()
        if len(layer_groups) != n_samples
    }
    if invalid_lengths:
        raise ValueError(
            f"groups must have length {n_samples} for every layer; got {invalid_lengths}"
        )

    unique_groups = sorted(
        {group for layer_groups in groups_by_layer.values() for group in layer_groups},
        key=lambda v: v.casefold(),
    )
    group_colors = _label_color_map(unique_groups)
    is_3d = n_components == 3

    def _make_trace(group: str, coords: torch.Tensor, layer: int):
        layer_groups = groups_by_layer[layer]
        indices = [i for i, v in enumerate(layer_groups) if v == group]
        return _embedding_trace(
            coords,
            indices,
            n_components=n_components,
            name=group,
            marker=dict(
                size=5 if is_3d else 9, opacity=0.82, color=group_colors[group]
            ),
            text=[samples.hover_text[i] for i in indices],
            hovertemplate=_embedding_hovertemplate(
                group, x_label, y_label, z_label if is_3d else None
            ),
        )

    first_layer = selected_layers[0]
    traces = [
        _make_trace(g, layer_coords[first_layer], first_layer) for g in unique_groups
    ]
    frames = []
    for layer in selected_layers:
        coords = layer_coords[layer]
        ranges = layer_ranges[layer]
        if is_3d:
            x_range, y_range, z_range = ranges
            frame_layout = _layer_frame_layout(
                title, layer, x_range, y_range, z_range=z_range
            )
        else:
            x_range, y_range = ranges
            frame_layout = _layer_frame_layout(title, layer, x_range, y_range)
        data = [_make_trace(g, coords, layer) for g in unique_groups]
        frames.append(go.Frame(name=str(layer), data=data, layout=frame_layout))

    fig = go.Figure(data=traces, frames=frames)
    layout_kwargs = dict(
        title={
            "text": f"{title} - Layer {first_layer}",
            "font": {"size": 24},
            "y": 0.98,
            "yanchor": "top",
        },
        template="plotly_white",
        margin=dict(t=140, b=90),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02),
        updatemenus=_layer_animation_buttons(),
        sliders=_layer_slider(selected_layers),
    )
    if is_3d:
        first_x_range, first_y_range, first_z_range = layer_ranges[first_layer]
        layout_kwargs["scene"] = dict(
            xaxis=dict(title=x_label, range=first_x_range),
            yaxis=dict(title=y_label, range=first_y_range),
            zaxis=dict(title=z_label, range=first_z_range),
        )
        fig.update_layout(**layout_kwargs)
    else:
        layout_kwargs["xaxis_title"] = x_label
        layout_kwargs["yaxis_title"] = y_label
        fig.update_layout(**layout_kwargs)
        first_x_range, first_y_range = layer_ranges[first_layer]
        fig.update_xaxes(range=first_x_range, zeroline=True, automargin=True)
        fig.update_yaxes(range=first_y_range, zeroline=True, automargin=True)
    return fig


def _build_layered_similarity_figure(
    samples: LayeredSamples,
    selected_layers: list[int],
    title: str = "Centered Cosine Similarity by Layer",
    matrices: dict[int, np.ndarray] | None = None,
) -> go.Figure:
    if matrices is None:
        matrices = _layer_cosine_matrices(samples.vectors, selected_layers)
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
    return fig


def build_pair_similarity_figure(
    samples: LayeredSamples,
    layers: list[int] | None = None,
    title: str = "Centered Pair Similarity Across Layers",
) -> go.Figure:
    """Plot each persona-pair similarity as a line across layers."""

    selected_layers = _validate_layers(samples.vectors, layers)
    matrices = _layer_cosine_matrices(samples.vectors, selected_layers)
    return _build_pair_similarity_figure(samples, selected_layers, matrices, title)


def build_similarity_figures(
    samples: LayeredSamples,
    layers: list[int] | None = None,
    title: str = "Centered Cosine Similarity by Layer",
    pair_title: str = "Centered Pair Similarity Across Layers",
) -> tuple[go.Figure, go.Figure]:
    """Build similarity heatmap and pair trajectory figures from one matrix pass."""

    selected_layers = _validate_layers(samples.vectors, layers)
    matrices = _layer_cosine_matrices(samples.vectors, selected_layers)
    return (
        _build_layered_similarity_figure(samples, selected_layers, title, matrices),
        _build_pair_similarity_figure(samples, selected_layers, matrices, pair_title),
    )


def plot_hdbscan_cluster_counts(
    samples_by_variant: dict[str, LayeredSamples],
    *,
    min_cluster_size: int = 2,
    layers: list[int] | None = None,
    title: str = "HDBSCAN clusters by layer",
) -> go.Figure:
    """Per-layer HDBSCAN cluster count, one trace per variant.

    Hover also shows the number of points HDBSCAN labels as noise at that layer.
    """
    fig = go.Figure()
    colors = _label_color_map(list(samples_by_variant))
    for variant, samples in samples_by_variant.items():
        selected = _validate_layers(samples.vectors, layers)
        counts, noise = [], []
        for layer in selected:
            ids = cluster_hdbscan(
                samples.vectors[:, layer, :], min_cluster_size=min_cluster_size
            )
            counts.append(len(set(ids) - {-1}))
            noise.append(int((ids == -1).sum()))
        fig.add_trace(
            go.Scatter(
                x=selected,
                y=counts,
                mode="lines+markers",
                name=variant,
                line=dict(color=colors[variant]),
                customdata=noise,
                hovertemplate=(
                    f"{variant}<br>Layer %{{x}}<br>"
                    "Clusters: %{y}<br>Noise points: %{customdata}<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Layer",
        yaxis_title="Cluster count",
        template="plotly_white",
        hovermode="x",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02),
    )
    return fig


def plot_persona_dendrogram(
    samples: LayeredSamples,
    *,
    layer: int | None = None,
    layers: list[int] | None = None,
    layered: bool = False,
    linkage: str = "ward",
    center: bool = True,
    normalize: bool = True,
    title: str | None = None,
) -> go.Figure:
    """Hierarchical-linkage dendrogram over personas.

    Computed by default on the per-persona mean activation across all layers,
    matching the global clustering convention; pass ``layer`` for a single
    layer. Pass ``layered=True`` to build an interactive per-layer dendrogram
    with the shared layer slider/animation controls. Inputs are centered and
    L2-normalized by default, so distances mostly reflect vector direction
    rather than raw magnitude. Width auto-scales with the persona count so
    labels stay readable.
    """

    linkage = _validate_linkage(linkage)
    yaxis_title = _dendrogram_distance_label(linkage, normalize)
    if layer is not None and (layered or layers is not None):
        raise ValueError("Pass either layer or layered/layers, not both")

    if layered or layers is not None:
        selected_layers = _validate_layers(samples.vectors, layers)
        n = len(samples.labels)
        plot_title = title or _dendrogram_title(linkage)
        layer_figs = {
            selected_layer: _create_persona_dendrogram(
                samples.vectors[:, selected_layer, :],
                labels=samples.labels,
                linkage=linkage,
                center=center,
                normalize=normalize,
            )
            for selected_layer in selected_layers
        }
        first_layer = selected_layers[0]
        first_fig = layer_figs[first_layer]
        layer_y_ranges = {
            selected_layer: _trace_y_range(layer_fig.data)
            for selected_layer, layer_fig in layer_figs.items()
        }
        shared_y_range = [
            0.0,
            max(layer_range[1] for layer_range in layer_y_ranges.values()),
        ]

        frames = []
        for selected_layer in selected_layers:
            layer_fig = layer_figs[selected_layer]
            frame_layout = _layer_frame_layout(plot_title, selected_layer)
            frame_layout["xaxis"] = layer_fig.layout.xaxis.to_plotly_json()
            frame_layout["yaxis"] = layer_fig.layout.yaxis.to_plotly_json()
            frame_layout["yaxis"]["title"] = yaxis_title
            frame_layout["yaxis"]["range"] = shared_y_range
            frames.append(
                go.Frame(
                    name=str(selected_layer),
                    data=list(layer_fig.data),
                    layout=frame_layout,
                )
            )

        fig = go.Figure(data=list(first_fig.data), frames=frames)
        fig.update_layout(
            title={
                "text": f"{plot_title} - Layer {first_layer}",
                "font": {"size": 24},
                "y": 0.98,
                "yanchor": "top",
            },
            template="plotly_white",
            margin=dict(t=140, b=260),
            yaxis_title=yaxis_title,
            width=max(800, 18 * n),
            updatemenus=_layer_animation_buttons(),
            sliders=_layer_slider(selected_layers, pad_t=115),
        )
        fig.update_xaxes(
            **first_fig.layout.xaxis.to_plotly_json(),
            tickangle=-45,
            automargin=True,
        )
        fig.update_yaxes(
            **first_fig.layout.yaxis.to_plotly_json(),
            range=shared_y_range,
            automargin=True,
        )
        return fig

    if layer is None:
        data = prepare_layer_mean_cluster_samples(
            samples.vectors, center=center, normalize=normalize
        )
        suffix = "mean across layers"
        data_center = False
        data_normalize = False
    else:
        _validate_layers(samples.vectors, [layer])
        data = samples.vectors[:, layer, :]
        suffix = f"layer {layer}"
        data_center = center
        data_normalize = normalize

    n = len(samples.labels)
    fig = _create_persona_dendrogram(
        data,
        labels=samples.labels,
        linkage=linkage,
        center=data_center,
        normalize=data_normalize,
    )
    fig.update_layout(
        title=title or f"{_dendrogram_title(linkage)} - {suffix}",
        template="plotly_white",
        margin=dict(t=80, b=160),
        yaxis_title=yaxis_title,
        width=max(800, 18 * n),
    )
    fig.update_xaxes(tickangle=-45, automargin=True)
    fig.update_yaxes(automargin=True)
    return fig


def build_layered_figure(
    samples: LayeredSamples,
    kind: str,
    layers: list[int] | None = None,
    title: str | None = None,
    n_components: int = 2,
    n_clusters: int | None = None,
    cluster_seed: int = 0,
    cluster_mode: ClusterMode = "mean_across_layers",
    cluster_method: ClusterMethod = "kmeans",
    cluster_linkage: str = "ward",
    min_cluster_size: int = 2,
    min_samples: int | None = None,
    groups: list[str] | dict[int, list[str]] | None = None,
) -> go.Figure:
    """Build an interactive per-layer PCA, UMAP, or similarity figure.

    This is the main plotting entry point for persona-space views. It accepts
    the ``LayeredSamples`` returned by analysis helpers and adds the shared
    layer slider/animation controls used by all layered plots.

    For ``kind="pca"`` and ``kind="umap"``, ``n_components`` selects between a
    2D scatter (default) and a 3D scatter view. Two ways to override the
    default per-persona coloring (only valid for ``kind`` in
    ``{"pca", "umap"}``):

    - ``n_clusters=k``: convenience for clustering. ``cluster_method`` selects
      ``"kmeans"``, ``"agglomerative"``, or ``"hdbscan"``. ``cluster_mode``
      controls whether labels come from centered/unit per-layer means
      (``"mean_across_layers"``), the first selected layer (``"first_layer"``),
      or are recomputed independently for every frame (``"per_layer"``).
      ``n_clusters`` is required for k-means and agglomerative clustering;
      HDBSCAN uses ``min_cluster_size`` and labels outliers as ``"Noise"``.
    - ``groups``: a length-``n_samples`` list of group labels (e.g. produced
      by ``cluster_agglomerative`` or ``cluster_hdbscan``). Use this for any
      clustering method you want.

    ``n_clusters`` and ``groups`` are mutually exclusive.
    """

    selected_layers = _validate_layers(samples.vectors, layers)
    n_samples = samples.vectors.shape[0]
    if n_samples < 2:
        raise ValueError("At least two samples are required")

    if n_clusters is not None and groups is not None:
        raise ValueError("Pass either n_clusters or groups, not both")
    if n_clusters is not None or cluster_method == "hdbscan":
        if groups is not None:
            raise ValueError("Pass either clustering options or groups, not both")
        if kind == "similarity":
            raise ValueError(
                "groups/n_clusters are not supported for kind='similarity'"
            )
        if cluster_mode == "mean_across_layers":
            cluster_samples = prepare_layer_mean_cluster_samples(samples.vectors)
            groups = _cluster_projection_samples(
                cluster_samples,
                method=cluster_method,
                n_clusters=n_clusters,
                seed=cluster_seed,
                linkage=cluster_linkage,
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                center=False,
                normalize=False,
            )
        elif cluster_mode == "first_layer":
            groups = _cluster_projection_samples(
                samples.vectors[:, selected_layers[0], :],
                method=cluster_method,
                n_clusters=n_clusters,
                seed=cluster_seed,
                linkage=cluster_linkage,
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
            )
        elif cluster_mode == "per_layer":
            groups = {
                layer: _cluster_projection_samples(
                    samples.vectors[:, layer, :],
                    method=cluster_method,
                    n_clusters=n_clusters,
                    seed=cluster_seed,
                    linkage=cluster_linkage,
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                )
                for layer in selected_layers
            }
        else:
            raise ValueError(
                "cluster_mode must be one of: mean_across_layers, first_layer, per_layer"
            )
    if groups is not None:
        if kind == "similarity":
            raise ValueError(
                "groups/n_clusters are not supported for kind='similarity'"
            )
        if isinstance(groups, dict):
            invalid_lengths = {
                layer: len(layer_groups)
                for layer, layer_groups in groups.items()
                if len(layer_groups) != n_samples
            }
            if invalid_lengths:
                raise ValueError(
                    f"groups must have length {n_samples} for every layer; got {invalid_lengths}"
                )
        elif len(groups) != n_samples:
            raise ValueError(f"groups must have length {n_samples}; got {len(groups)}")

    if kind == "pca":
        return _build_layered_projection_figure(
            samples,
            selected_layers,
            title=title
            or ("PCA by Layer" if n_components == 2 else "PCA (3D) by Layer"),
            project_fn=project_pca,
            x_label="PC1",
            y_label="PC2",
            z_label="PC3" if n_components == 3 else None,
            n_components=n_components,
            groups=groups,
        )
    if kind == "umap":
        return _build_layered_projection_figure(
            samples,
            selected_layers,
            title=title
            or (
                "Centered UMAP by Layer"
                if n_components == 2
                else "Centered UMAP (3D) by Layer"
            ),
            project_fn=project_umap,
            x_label="UMAP 1",
            y_label="UMAP 2",
            z_label="UMAP 3" if n_components == 3 else None,
            n_components=n_components,
            groups=groups,
        )
    if kind == "similarity":
        return _build_layered_similarity_figure(
            samples,
            selected_layers,
            title=title or "Centered Cosine Similarity by Layer",
        )
    raise ValueError("kind must be one of: pca, umap, similarity")
