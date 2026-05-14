"""Hierarchical-linkage dendrogram for personas."""

import numpy as np
import plotly.graph_objects as go
import torch

from persona_vectors.analysis import (
    LayeredSamples,
    prepare_cluster_samples,
    prepare_layer_mean_cluster_samples,
)
from persona_vectors.plots._common import (
    layer_animation_buttons,
    layer_frame_layout,
    layer_slider,
    validate_layers,
)


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
        selected_layers = validate_layers(samples.vectors, layers)
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
            frame_layout = layer_frame_layout(plot_title, selected_layer)
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
            updatemenus=layer_animation_buttons(),
            sliders=layer_slider(selected_layers, pad_t=115),
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
        validate_layers(samples.vectors, [layer])
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
