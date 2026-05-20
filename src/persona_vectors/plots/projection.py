"""PCA / UMAP / Isomap projection figures with optional k-means coloring."""

from dataclasses import dataclass
from typing import Callable, Literal, NamedTuple

import plotly.graph_objects as go
import torch

from persona_vectors.analysis import (
    LayeredSamples,
    cluster_kmeans,
    prepare_cluster_samples,
    prepare_layer_mean_cluster_samples,
    project_isomap,
    project_pca,
    project_umap,
)
from persona_vectors.plots._common import (
    apply_fig_fonts,
    coordinate_range,
    label_color_map,
    layer_animation_buttons,
    layer_frame_layout,
    layer_slider,
    validate_layers,
)

_MAX_GROUP_LEGEND_TRACES = 40

ClusterMode = Literal["mean_across_layers", "first_layer", "per_layer"]
ProjectionKind = Literal["pca", "umap", "isomap"]


@dataclass(frozen=True)
class LayeredProjectionData:
    """Precomputed projection coordinates and optional graph edges by layer."""

    kind: ProjectionKind
    layers: tuple[int, ...]
    n_components: int
    normalize: bool
    graph_n_neighbors: int
    layer_coords: dict[int, torch.Tensor]
    layer_ranges: dict[int, tuple[tuple[float, float], ...]]
    graph_edges: dict[int, list[tuple[int, int]]]


class _NumericColoring(NamedTuple):
    label: str
    values_by_layer: dict[int, list[float]]
    colorscale: str
    value_min: float
    value_max: float
    colorbar: dict | None = None


class _CategoricalColoring(NamedTuple):
    label: str
    groups_by_layer: dict[int, list[str]]
    group_colors: dict[str, str]
    unique_groups: list[str]
    use_single_group_trace: bool = False


_Coloring = _NumericColoring | _CategoricalColoring


def _cluster_label(cluster_id: int) -> str:
    return f"Cluster {int(cluster_id)}"


def _cluster_projection_samples(
    samples: torch.Tensor,
    *,
    n_clusters: int | None,
    seed: int,
    center: bool = True,
    normalize: bool = True,
) -> list[str]:
    if n_clusters is None:
        raise ValueError("n_clusters is required for clustering")
    cluster_ids = cluster_kmeans(
        samples,
        n_clusters=n_clusters,
        seed=seed,
        center=center,
        normalize=normalize,
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
    customdata: list[str] | None = None,
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

    kwargs["mode"] = "markers"
    if name is not None:
        kwargs["name"] = name
    if marker is not None:
        kwargs["marker"] = marker
    if text is not None:
        kwargs["text"] = text
    if customdata is not None:
        kwargs["customdata"] = customdata
    if hovertemplate is not None:
        kwargs["hovertemplate"] = hovertemplate
    return trace_cls(**kwargs)


def _projection_graph_edges(
    samples: torch.Tensor,
    *,
    n_neighbors: int,
) -> list[tuple[int, int]]:
    from sklearn.neighbors import NearestNeighbors

    n_samples = int(samples.shape[0])
    if n_samples < 2:
        return []
    n_neighbors = min(max(1, n_neighbors), n_samples - 1)
    prepared = prepare_cluster_samples(samples, center=True, normalize=True)
    neighbors = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean")
    neighbors.fit(prepared.cpu().numpy())
    indices = neighbors.kneighbors(return_distance=False)[:, 1:]

    edges = set()
    for left, row in enumerate(indices):
        for right in row.tolist():
            edges.add(tuple(sorted((left, int(right)))))
    return sorted(edges)


def _graph_edge_trace(
    coords: torch.Tensor,
    edges: list[tuple[int, int]],
    *,
    n_components: int,
) -> go.Scattergl | go.Scatter3d:
    x_values: list[float | None] = []
    y_values: list[float | None] = []
    z_values: list[float | None] = []
    for left, right in edges:
        x_values.extend([float(coords[left, 0]), float(coords[right, 0]), None])
        y_values.extend([float(coords[left, 1]), float(coords[right, 1]), None])
        if n_components == 3:
            z_values.extend([float(coords[left, 2]), float(coords[right, 2]), None])

    trace_kwargs = dict(
        x=x_values,
        y=y_values,
        mode="lines",
        name="kNN graph",
        hoverinfo="skip",
        showlegend=False,
        line=dict(color="rgba(71, 85, 105, 0.24)", width=1),
    )
    if n_components == 3:
        trace_kwargs["z"] = z_values
        return go.Scatter3d(**trace_kwargs)
    return go.Scattergl(**trace_kwargs)


def _validate_color_values(
    color_values: list[float] | dict[int, list[float]],
    selected_layers: list[int],
    n_samples: int,
) -> dict[int, list[float]]:
    if isinstance(color_values, dict):
        missing = [layer for layer in selected_layers if layer not in color_values]
        if missing:
            raise ValueError(f"color_values is missing layer(s): {missing}")
        values_by_layer = {
            layer: [float(value) for value in color_values[layer]]
            for layer in selected_layers
        }
    else:
        stable_values = [float(value) for value in color_values]
        values_by_layer = {layer: stable_values for layer in selected_layers}

    invalid_lengths = {
        layer: len(values)
        for layer, values in values_by_layer.items()
        if len(values) != n_samples
    }
    if invalid_lengths:
        raise ValueError(
            f"color_values must have length {n_samples} for every layer; got {invalid_lengths}"
        )
    return values_by_layer


def _prepare_layered_projection_data(
    samples: LayeredSamples,
    kind: ProjectionKind,
    selected_layers: list[int],
    *,
    project_fn: Callable[..., torch.Tensor],
    n_components: int,
    normalize: bool,
    graph_overlay: bool,
    graph_n_neighbors: int,
    project_kwargs: dict | None = None,
) -> LayeredProjectionData:
    if n_components not in (2, 3):
        raise ValueError("n_components must be 2 or 3")

    layer_inputs = [samples.vectors[:, layer, :] for layer in selected_layers]
    project_kwargs = {} if project_kwargs is None else project_kwargs
    coords_list = [
        project_fn(layer_input, n_components=n_components, **project_kwargs)
        for layer_input in layer_inputs
    ]
    layer_coords = dict(zip(selected_layers, coords_list))
    layer_ranges = {
        layer: tuple(coordinate_range(coords, axis) for axis in range(n_components))
        for layer, coords in layer_coords.items()
    }
    graph_edges = (
        {
            layer: _projection_graph_edges(
                samples.vectors[:, layer, :],
                n_neighbors=graph_n_neighbors,
            )
            for layer in selected_layers
        }
        if graph_overlay
        else {}
    )
    return LayeredProjectionData(
        kind=kind,
        layers=tuple(selected_layers),
        n_components=n_components,
        normalize=normalize,
        graph_n_neighbors=graph_n_neighbors,
        layer_coords=layer_coords,
        layer_ranges=layer_ranges,
        graph_edges=graph_edges,
    )


def _validate_projection_data(
    projection_data: LayeredProjectionData,
    *,
    kind: ProjectionKind,
    selected_layers: list[int],
    n_components: int,
    normalize: bool,
    graph_overlay: bool,
    graph_n_neighbors: int,
) -> None:
    if projection_data.kind != kind:
        raise ValueError(
            "projection_data kind must match the requested kind; "
            f"got {projection_data.kind!r} and {kind!r}"
        )
    if projection_data.layers != tuple(selected_layers):
        raise ValueError(
            "projection_data layers must match the requested layers; "
            f"got {list(projection_data.layers)} and {selected_layers}"
        )
    if projection_data.n_components != n_components:
        raise ValueError(
            "projection_data n_components must match the requested n_components; "
            f"got {projection_data.n_components} and {n_components}"
        )
    if projection_data.normalize != normalize:
        raise ValueError(
            "projection_data normalize must match the requested normalize setting; "
            f"got {projection_data.normalize} and {normalize}"
        )
    if projection_data.graph_n_neighbors != graph_n_neighbors:
        raise ValueError(
            "projection_data graph_n_neighbors must match the requested graph_n_neighbors; "
            f"got {projection_data.graph_n_neighbors} and {graph_n_neighbors}"
        )
    if graph_overlay:
        missing_graph_layers = [
            layer
            for layer in selected_layers
            if layer not in projection_data.graph_edges
        ]
        if missing_graph_layers:
            raise ValueError(
                "projection_data was prepared without graph edges for layer(s): "
                f"{missing_graph_layers}"
            )


def projection_spec(
    kind: ProjectionKind,
    n_components: int,
    graph_n_neighbors: int,
    normalize: bool = True,
) -> tuple[str, Callable[..., torch.Tensor], str, str, str | None, dict | None]:
    if kind == "pca":
        return (
            "PCA by Layer" if n_components == 2 else "PCA (3D) by Layer",
            project_pca,
            "PC1",
            "PC2",
            "PC3" if n_components == 3 else None,
            {"normalize": normalize},
        )
    if kind == "umap":
        return (
            (
                "Centered UMAP by Layer"
                if n_components == 2
                else "Centered UMAP (3D) by Layer"
            ),
            project_umap,
            "UMAP 1",
            "UMAP 2",
            "UMAP 3" if n_components == 3 else None,
            {"normalize": normalize},
        )
    if kind == "isomap":
        return (
            (
                "Centered Isomap by Layer"
                if n_components == 2
                else "Centered Isomap (3D) by Layer"
            ),
            project_isomap,
            "Isomap 1",
            "Isomap 2",
            "Isomap 3" if n_components == 3 else None,
            {"n_neighbors": graph_n_neighbors},
        )
    raise ValueError("kind must be one of: pca, umap, isomap")


def prepare_layered_projection_data(
    samples: LayeredSamples,
    kind: ProjectionKind,
    layers: list[int] | None = None,
    n_components: int = 2,
    normalize: bool = True,
    graph_overlay: bool = False,
    graph_n_neighbors: int = 5,
) -> LayeredProjectionData:
    """Precompute layered projection coordinates for repeated figure coloring.

    Use this when the same PCA/UMAP/Isomap layout will be redrawn with
    different ``groups`` or ``color_values``. The result is independent of
    coloring and can be passed to ``build_layered_figure(..., projection_data=...)``.
    """
    selected_layers = validate_layers(samples.vectors, layers)
    n_samples = samples.vectors.shape[0]
    if n_samples < 2:
        raise ValueError("At least two samples are required")
    effective_normalize = normalize if kind in ("pca", "umap") else True
    _, project_fn, _, _, _, project_kwargs = projection_spec(
        kind, n_components, graph_n_neighbors, normalize=effective_normalize
    )
    return _prepare_layered_projection_data(
        samples,
        kind,
        selected_layers,
        project_fn=project_fn,
        n_components=n_components,
        normalize=effective_normalize,
        graph_overlay=graph_overlay,
        graph_n_neighbors=graph_n_neighbors,
        project_kwargs=project_kwargs,
    )


def prepare_kmeans_groups(
    samples: LayeredSamples,
    *,
    layers: list[int] | None = None,
    n_clusters: int,
    cluster_seed: int = 0,
    cluster_mode: ClusterMode = "mean_across_layers",
) -> list[str] | dict[int, list[str]]:
    """Precompute k-means group labels for projection coloring.

    The labels are independent of PCA/UMAP/Isomap coordinates, so UI callers can
    cache them separately from projection data and reuse them across redraws.
    """
    selected_layers = validate_layers(samples.vectors, layers)
    if cluster_mode == "mean_across_layers":
        cluster_samples = prepare_layer_mean_cluster_samples(samples.vectors)
        return _cluster_projection_samples(
            cluster_samples,
            n_clusters=n_clusters,
            seed=cluster_seed,
            center=False,
            normalize=False,
        )
    if cluster_mode == "first_layer":
        return _cluster_projection_samples(
            samples.vectors[:, selected_layers[0], :],
            n_clusters=n_clusters,
            seed=cluster_seed,
        )
    if cluster_mode == "per_layer":
        return {
            layer: _cluster_projection_samples(
                samples.vectors[:, layer, :],
                n_clusters=n_clusters,
                seed=cluster_seed,
            )
            for layer in selected_layers
        }
    raise ValueError(
        "cluster_mode must be one of: mean_across_layers, first_layer, per_layer"
    )


def _projection_coloring(
    samples: LayeredSamples,
    selected_layers: list[int],
    *,
    groups: list[str] | dict[int, list[str]] | None,
    color_values: list[float] | dict[int, list[float]] | None,
    color_label: str,
    colorscale: str,
    color_tickvals: list[float] | None,
    color_ticktext: list[str] | None,
) -> _Coloring:
    n_samples = int(samples.vectors.shape[0])
    if groups is not None and color_values is not None:
        raise ValueError("Pass either groups or color_values, not both")

    if color_values is not None:
        values_by_layer = _validate_color_values(
            color_values, selected_layers, n_samples
        )
        all_values = [
            value for layer_values in values_by_layer.values() for value in layer_values
        ]
        colorbar = dict(title=color_label)
        if color_tickvals is not None:
            colorbar["tickvals"] = color_tickvals
        if color_ticktext is not None:
            colorbar["ticktext"] = color_ticktext
        return _NumericColoring(
            label=color_label,
            values_by_layer=values_by_layer,
            colorscale=colorscale,
            value_min=min(all_values),
            value_max=max(all_values),
            colorbar=colorbar,
        )

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
    return _CategoricalColoring(
        label="Personas" if groups is None else "Groups",
        groups_by_layer=groups_by_layer,
        group_colors=label_color_map(unique_groups),
        unique_groups=unique_groups,
        use_single_group_trace=len(unique_groups) > _MAX_GROUP_LEGEND_TRACES,
    )


def _projection_layer_traces(
    samples: LayeredSamples,
    coords: torch.Tensor,
    layer: int,
    coloring: _Coloring,
    *,
    n_components: int,
    x_label: str,
    y_label: str,
    z_label: str | None,
) -> list[go.Scattergl | go.Scatter3d]:
    n_samples = int(samples.vectors.shape[0])
    is_3d = n_components == 3
    marker_size = 5 if is_3d else 9

    if isinstance(coloring, _NumericColoring):
        values = coloring.values_by_layer[layer]
        marker = dict(
            size=marker_size,
            opacity=0.82,
            color=values,
            colorscale=coloring.colorscale,
            cmin=coloring.value_min,
            cmax=coloring.value_max,
            colorbar=coloring.colorbar,
        )
        return [
            _embedding_trace(
                coords,
                list(range(n_samples)),
                n_components=n_components,
                name=coloring.label,
                marker=marker,
                text=samples.hover_text,
                hovertemplate=(
                    "%{text}<br>"
                    + coloring.label
                    + ": %{marker.color}<br>"
                    + f"{x_label}=%{{x:.4f}}<br>"
                    + f"{y_label}=%{{y:.4f}}"
                    + (
                        f"<br>{z_label}=%{{z:.4f}}"
                        if is_3d and z_label is not None
                        else ""
                    )
                    + "<extra></extra>"
                ),
            )
        ]

    layer_groups = coloring.groups_by_layer[layer]
    if coloring.use_single_group_trace:
        return [
            _embedding_trace(
                coords,
                list(range(n_samples)),
                n_components=n_components,
                name=coloring.label,
                marker=dict(
                    size=marker_size,
                    opacity=0.82,
                    color=[coloring.group_colors[group] for group in layer_groups],
                ),
                text=samples.hover_text,
                customdata=layer_groups,
                hovertemplate=(
                    "%{text}<br>Group: %{customdata}<br>"
                    + f"{x_label}=%{{x:.4f}}<br>"
                    + f"{y_label}=%{{y:.4f}}"
                    + (
                        f"<br>{z_label}=%{{z:.4f}}"
                        if is_3d and z_label is not None
                        else ""
                    )
                    + "<extra></extra>"
                ),
            )
        ]

    traces = []
    for group in coloring.unique_groups:
        indices = [i for i, value in enumerate(layer_groups) if value == group]
        traces.append(
            _embedding_trace(
                coords,
                indices,
                n_components=n_components,
                name=group,
                marker=dict(
                    size=marker_size,
                    opacity=0.82,
                    color=coloring.group_colors[group],
                ),
                text=[samples.hover_text[i] for i in indices],
                hovertemplate=_embedding_hovertemplate(
                    group, x_label, y_label, z_label if is_3d else None
                ),
            )
        )
    return traces


def _projection_frame_layout(
    title: str,
    layer: int,
    ranges: tuple[tuple[float, float], ...],
    *,
    is_3d: bool,
) -> dict:
    if is_3d:
        x_range, y_range, z_range = ranges
        return layer_frame_layout(title, layer, x_range, y_range, z_range=z_range)
    x_range, y_range = ranges
    return layer_frame_layout(title, layer, x_range, y_range)


def build_layered_projection_figure(
    samples: LayeredSamples,
    selected_layers: list[int],
    kind: ProjectionKind,
    title: str,
    project_fn,
    x_label: str,
    y_label: str,
    z_label: str | None = None,
    n_components: int = 2,
    groups: list[str] | dict[int, list[str]] | None = None,
    graph_overlay: bool = False,
    graph_n_neighbors: int = 5,
    color_values: list[float] | dict[int, list[float]] | None = None,
    color_label: str = "Value",
    colorscale: str = "Viridis",
    color_tickvals: list[float] | None = None,
    color_ticktext: list[str] | None = None,
    project_kwargs: dict | None = None,
    projection_data: LayeredProjectionData | None = None,
    normalize: bool = True,
) -> go.Figure:
    if n_components not in (2, 3):
        raise ValueError("n_components must be 2 or 3")
    effective_normalize = normalize if kind in ("pca", "umap") else True

    if projection_data is None:
        projection_data = _prepare_layered_projection_data(
            samples,
            kind,
            selected_layers,
            project_fn=project_fn,
            n_components=n_components,
            normalize=effective_normalize,
            graph_overlay=graph_overlay,
            graph_n_neighbors=graph_n_neighbors,
            project_kwargs=project_kwargs,
        )
    else:
        _validate_projection_data(
            projection_data,
            kind=kind,
            selected_layers=selected_layers,
            n_components=n_components,
            normalize=effective_normalize,
            graph_overlay=graph_overlay,
            graph_n_neighbors=graph_n_neighbors,
        )
    layer_coords = projection_data.layer_coords
    layer_ranges = projection_data.layer_ranges
    graph_edges = projection_data.graph_edges

    coloring = _projection_coloring(
        samples,
        selected_layers,
        groups=groups,
        color_values=color_values,
        color_label=color_label,
        colorscale=colorscale,
        color_tickvals=color_tickvals,
        color_ticktext=color_ticktext,
    )
    is_3d = n_components == 3

    def _layer_traces(coords: torch.Tensor, layer: int):
        traces = []
        if graph_overlay:
            traces.append(
                _graph_edge_trace(
                    coords,
                    graph_edges[layer],
                    n_components=n_components,
                )
            )
        traces.extend(
            _projection_layer_traces(
                samples,
                coords,
                layer,
                coloring,
                n_components=n_components,
                x_label=x_label,
                y_label=y_label,
                z_label=z_label,
            )
        )
        return traces

    first_layer = selected_layers[0]
    traces = _layer_traces(layer_coords[first_layer], first_layer)
    frames = [
        go.Frame(
            name=str(layer),
            data=_layer_traces(layer_coords[layer], layer),
            layout=_projection_frame_layout(
                title,
                layer,
                layer_ranges[layer],
                is_3d=is_3d,
            ),
        )
        for layer in selected_layers
    ]

    fig = go.Figure(data=traces, frames=frames)
    return _apply_layered_projection_layout(
        fig,
        title,
        selected_layers,
        layer_ranges,
        x_label,
        y_label,
        z_label,
        n_components,
    )


def _apply_layered_projection_layout(
    fig: go.Figure,
    title: str,
    selected_layers: list[int],
    layer_ranges: dict[int, tuple[tuple[float, float], ...]],
    x_label: str,
    y_label: str,
    z_label: str | None,
    n_components: int,
) -> go.Figure:
    first_layer = selected_layers[0]
    is_3d = n_components == 3
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
        updatemenus=layer_animation_buttons(),
        sliders=layer_slider(selected_layers),
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
    return apply_fig_fonts(fig)
