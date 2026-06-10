"""Plotting helpers for persona vectors.

This package is split by view: the public API is re-exported here so
callers can keep using ``from persona_vectors.plots import ...``.
"""

from persona_vectors.plots._common import save_plot_html
from persona_vectors.plots.builder import build_layered_figure
from persona_vectors.plots.correlations import build_cooccurrence_heatmap
from persona_vectors.plots.dendrogram import plot_persona_dendrogram
from persona_vectors.plots.probes import (
    plot_attribute_layer_selectivity_heatmap,
    plot_metric_comparison,
    plot_metric_over_layers,
)
from persona_vectors.plots.projection import (
    ClusterMode,
    LayeredProjectionData,
    ProjectionKind,
    prepare_kmeans_groups,
    prepare_layered_projection_data,
)
from persona_vectors.plots.scree import plot_scree
from persona_vectors.plots.similarity import (
    build_pair_similarity_figure,
    build_similarity_figures,
    plot_layer_similarity,
)

__all__ = [
    "ClusterMode",
    "LayeredProjectionData",
    "ProjectionKind",
    "build_cooccurrence_heatmap",
    "build_layered_figure",
    "build_pair_similarity_figure",
    "build_similarity_figures",
    "plot_attribute_layer_selectivity_heatmap",
    "plot_layer_similarity",
    "plot_metric_comparison",
    "plot_metric_over_layers",
    "plot_persona_dendrogram",
    "plot_scree",
    "prepare_kmeans_groups",
    "prepare_layered_projection_data",
    "save_plot_html",
]
