"""``build_layered_figure``: top-level dispatcher for layered persona views."""

import plotly.graph_objects as go

from persona_vectors.analysis import LayeredSamples
from persona_vectors.plots._common import validate_layers
from persona_vectors.plots.projection import (
    ClusterMode,
    LayeredProjectionData,
    build_layered_projection_figure,
    prepare_kmeans_groups,
    projection_spec,
)
from persona_vectors.plots.similarity import build_layered_similarity_figure


def build_layered_figure(
    samples: LayeredSamples,
    kind: str,
    layers: list[int] | None = None,
    title: str | None = None,
    n_components: int = 2,
    n_clusters: int | None = None,
    cluster_seed: int = 0,
    cluster_mode: ClusterMode = "mean_across_layers",
    groups: list[str] | dict[int, list[str]] | None = None,
    graph_overlay: bool = False,
    graph_n_neighbors: int = 5,
    projection_normalize: bool = True,
    color_values: list[float] | dict[int, list[float]] | None = None,
    color_label: str = "Value",
    colorscale: str = "Viridis",
    color_tickvals: list[float] | None = None,
    color_ticktext: list[str] | None = None,
    projection_data: LayeredProjectionData | None = None,
) -> go.Figure:
    """Build an interactive per-layer PCA, UMAP, Isomap, or similarity figure.

    This is the main plotting entry point for persona-space views. It accepts
    the ``LayeredSamples`` returned by analysis helpers and adds the shared
    layer slider/animation controls used by all layered plots.

    For projection kinds, ``n_components`` selects between a 2D scatter
    (default) and a 3D scatter view. Options for overriding the default
    per-persona coloring or adding context:

    - ``n_clusters=k``: convenience for k-means clustering. ``cluster_mode``
      controls whether labels come from centered/unit per-layer means
      (``"mean_across_layers"``), the first selected layer
      (``"first_layer"``), or are recomputed independently for every frame
      (``"per_layer"``).
    - ``groups``: a length-``n_samples`` list of group labels (e.g. produced
      elsewhere). Use this for any categorical grouping.
    - ``color_values``: a numeric length-``n_samples`` list for continuous or
      ordinal color scales. ``color_label``, ``colorscale``, ``color_tickvals``,
      and ``color_ticktext`` control the colorbar.
    - ``graph_overlay=True``: draw the centered/unit-vector kNN graph behind
      projection points. This is most useful for Isomap.
    - ``projection_normalize``: center and L2-normalize persona vectors before
      PCA/UMAP projection. Enabled by default to match cosine, Isomap, and
      k-means geometry.
    - ``projection_data``: precomputed coordinates returned by
      ``prepare_layered_projection_data``. This lets callers redraw the same
      projection with different colors without recomputing PCA/UMAP/Isomap.

    ``n_clusters``, ``groups``, and ``color_values`` are mutually exclusive.
    """
    selected_layers = validate_layers(samples.vectors, layers)
    n_samples = samples.vectors.shape[0]
    if n_samples < 2:
        raise ValueError("At least two samples are required")

    if sum(opt is not None for opt in (n_clusters, groups, color_values)) > 1:
        raise ValueError("n_clusters, groups, and color_values are mutually exclusive")

    if kind == "similarity" and (
        n_clusters is not None or groups is not None or color_values is not None
    ):
        raise ValueError(
            "n_clusters, groups, and color_values are not supported for kind='similarity'"
        )

    if n_clusters is not None:
        groups = prepare_kmeans_groups(
            samples,
            layers=selected_layers,
            n_clusters=n_clusters,
            cluster_seed=cluster_seed,
            cluster_mode=cluster_mode,
        )

    if kind in ("pca", "umap", "isomap"):
        effective_projection_normalize = (
            projection_normalize if kind in ("pca", "umap") else True
        )
        default_title, project_fn, x_label, y_label, z_label, project_kwargs = (
            projection_spec(
                kind,
                n_components,
                graph_n_neighbors,
                normalize=effective_projection_normalize,
            )
        )
        return build_layered_projection_figure(
            samples,
            selected_layers,
            kind,
            title=title or default_title,
            project_fn=project_fn,
            x_label=x_label,
            y_label=y_label,
            z_label=z_label,
            n_components=n_components,
            groups=groups,
            graph_overlay=graph_overlay,
            graph_n_neighbors=graph_n_neighbors,
            color_values=color_values,
            color_label=color_label,
            colorscale=colorscale,
            color_tickvals=color_tickvals,
            color_ticktext=color_ticktext,
            project_kwargs=project_kwargs,
            projection_data=projection_data,
            normalize=effective_projection_normalize,
        )
    if kind == "similarity":
        if projection_data is not None:
            raise ValueError("projection_data is not supported for kind='similarity'")
        return build_layered_similarity_figure(
            samples,
            selected_layers,
            title=title or "Centered Cosine Similarity by Layer",
        )
    raise ValueError("kind must be one of: pca, umap, isomap, similarity")
