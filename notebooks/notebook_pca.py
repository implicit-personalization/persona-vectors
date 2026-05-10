#!/usr/bin/env python

"""PCA and clustering views over persona vectors from the Hub or local artifacts.

For pairwise similarity, dendrograms, and prompt-variant comparisons see
``notebook_similarity.py``.
"""

# %% Imports

import torch
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from persona_vectors.analysis import (
    cluster_agglomerative_ward,
    cluster_hdbscan,
    list_comparison_personas,
    load_persona_vectors,
    pca_explained_variance,
)
from persona_vectors.artifacts import HFActivationStore
from persona_vectors.extraction import MaskStrategy
from persona_vectors.plots import (
    build_layered_figure,
    plot_hdbscan_cluster_counts,
    plot_scree,
)

console = Console()

# %% Setup
load_dotenv()
torch.set_grad_enabled(False)

REPO_ID = "implicit-personalization/synth-persona-vectors"
MODEL_NAME = "google/gemma-2-9b-it"
MASK_STRATEGY = MaskStrategy.ANSWER_MEAN
VARIANTS = ["biography", "templated"]
INCLUDE_BASELINE = False

# %% Load activation store
# Default: read the published Hub artifact dataset. To use local artifacts
# instead, comment out HFActivationStore and uncomment the two local lines.
store = HFActivationStore(REPO_ID, MODEL_NAME, mask_strategy=MASK_STRATEGY)
# from persona_vectors.artifacts import ActivationStore
# store = ActivationStore(MODEL_NAME, mask_strategy=MASK_STRATEGY)

available_variants = store.available_variants(VARIANTS)
comparison_variants = [variant for variant in VARIANTS if variant in available_variants]
persona_ids = list_comparison_personas(
    store,
    comparison_variants,
    include_baseline=INCLUDE_BASELINE,
)

summary = Table(title="Activation Dataset")
summary.add_column("Property", style="cyan")
summary.add_column("Value", style="magenta")
summary.add_row("Store", type(store).__name__)
summary.add_row("Repo", getattr(store, "repo_id", "local artifacts"))
summary.add_row("Model", store.model_name)
summary.add_row("Config", getattr(store, "config_name", str(MASK_STRATEGY)))
summary.add_row("Available variants", ", ".join(available_variants))
summary.add_row("Compared variants", ", ".join(comparison_variants))
summary.add_row("Personas loaded", str(len(persona_ids)))
console.print(summary)

# %% Load persona vectors for each variant
samples = {
    variant: load_persona_vectors(store, variant, persona_ids=persona_ids)
    for variant in comparison_variants
}

# %% Scree plot - PCA explained variance for representative layers
# NOTE: Usually the first 5-6 components carry most of the visible structure.
for variant, s in samples.items():
    num_layers = int(s.vectors.shape[1])
    scree_layers = sorted({0, num_layers // 3, (2 * num_layers) // 3, num_layers - 1})
    plot_scree(
        {
            f"layer {layer}": pca_explained_variance(s.vectors[:, layer, :])
            for layer in scree_layers
        },
        title=f"PCA explained variance - {variant} persona vectors",
        show=True,
    )

# %% PCA (3D) - layered view per variant, colored by k-means (k-means++) clusters
# Tweak N_CLUSTERS for your persona set. Clusters are fit once on the per-persona
# mean across layers so each persona keeps one stable color across every frame
# (per-layer re-clustering would shuffle colors as you slide).
# For the 2D version, drop n_components=3 (2D is the default).
N_CLUSTERS = 5
for variant, s in samples.items():
    build_layered_figure(
        s,
        "pca",
        title=f"PCA (3D) - {variant} persona vectors (k-means, k={N_CLUSTERS})",
        n_components=3,
        n_clusters=N_CLUSTERS,
    ).show()

# %% PCA (3D) - colored by Agglomerative (Ward) hierarchical clusters
# Same per-persona mean across layers as k-means, but using Ward linkage so
# clusters are formed by repeatedly merging pairs that minimize within-cluster
# variance. Still requires choosing k.
WARD_N_CLUSTERS = 5
for variant, s in samples.items():
    cluster_ids = cluster_agglomerative_ward(
        s.vectors.mean(dim=1), n_clusters=WARD_N_CLUSTERS
    )
    groups = [f"Cluster {c}" for c in cluster_ids]
    build_layered_figure(
        s,
        "pca",
        title=f"PCA (3D) - {variant} persona vectors (Ward, k={WARD_N_CLUSTERS})",
        n_components=3,
        groups=groups,
    ).show()

# %% PCA (3D) - colored by HDBSCAN (no k required; outliers labeled "Noise")
# HDBSCAN picks cluster counts from data density. With small persona sets keep
# MIN_CLUSTER_SIZE low (2-3); raise it as the persona count grows. Personas
# that don't belong to any dense region are tagged "Noise".
MIN_CLUSTER_SIZE = 2
for variant, s in samples.items():
    cluster_ids = cluster_hdbscan(
        s.vectors.mean(dim=1), min_cluster_size=MIN_CLUSTER_SIZE
    )
    groups = ["Noise" if c == -1 else f"Cluster {c}" for c in cluster_ids]
    build_layered_figure(
        s,
        "pca",
        title=f"PCA (3D) - {variant} persona vectors (HDBSCAN, min_cluster_size={MIN_CLUSTER_SIZE})",
        n_components=3,
        groups=groups,
    ).show()

# %% HDBSCAN cluster count by layer - does the count change across depth?
# Re-runs HDBSCAN on each layer's activations and plots the cluster count
# (excluding noise). Hover shows how many points were tagged as noise.
# plot_hdbscan_cluster_counts(samples, min_cluster_size=MIN_CLUSTER_SIZE).show()
