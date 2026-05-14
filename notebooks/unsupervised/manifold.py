#!/usr/bin/env python

"""Manifold and attribute views over persona vectors from Hub or local artifacts.

For pairwise similarity, dendrograms, and prompt-variant comparisons see
``notebook_similarity.py``.
"""

# %% Imports

import torch
from dotenv import load_dotenv
from persona_data.synth_persona import SynthPersonaDataset
from rich.console import Console
from rich.table import Table

from persona_vectors.analysis import (
    laplacian_eigenvalues,
    load_persona_vectors,
    pca_explained_variance,
)
from persona_vectors.artifacts import HFPersonaVectorStore
from persona_vectors.attributes import attribute_color_kwargs
from persona_vectors.extraction import MaskStrategy
from persona_vectors.plots import (
    build_layered_figure,
    plot_laplacian_eigengap,
    plot_scree,
)

console = Console()

# %% Setup
load_dotenv()
torch.set_grad_enabled(False)

REPO_ID = "implicit-personalization/synth-persona-vectors"
MODEL_NAME = "google/gemma-3-27b-it"
MASK_STRATEGY = MaskStrategy.ANSWER_MEAN
VARIANTS = ["biography", "templated"]
INCLUDE_BASELINE = False

# %% Load activation store
# Default: read the published Hub artifact dataset. To use local artifacts
# instead, comment out HFPersonaVectorStore and uncomment the two local lines.
store = HFPersonaVectorStore(REPO_ID, MODEL_NAME, mask_strategy=MASK_STRATEGY)

# NOTE: This is an example of how you may get things from another position locally

# MODEL_NAME = "meta-llama/Llama-3.1-405B-Instruct"
# store = PersonaVectorStore(
#     MODEL_NAME, mask_strategy=MASK_STRATEGY, root_dir="artifacts/persona-vectors"
# )

available_variants = store.available_variants(VARIANTS)
comparison_variants = [variant for variant in VARIANTS if variant in available_variants]
persona_ids = store.list_personas(
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

# %% Laplacian eigengap - suggested spectral cluster count per layer
# The largest gap between successive eigenvalues hints at the natural number
# of persona clusters (the spectral analogue of a scree elbow).
# NOTE: To review on how to get a meaningful number of eingavalues even in the smooth case
# Since this might not be super representative in our case
for variant, s in samples.items():
    num_layers = int(s.vectors.shape[1])
    eigen_layers = sorted({0, num_layers // 3, (2 * num_layers) // 3, num_layers - 1})
    plot_laplacian_eigengap(
        {
            f"layer {layer}": laplacian_eigenvalues(s.vectors[:, layer, :])
            for layer in eigen_layers
        },
        title=f"Laplacian eigengap - {variant} persona vectors",
        show=True,
    )

# %% Attribute schema overview
persona_dataset = SynthPersonaDataset()

attribute_summary = Table(title="Persona Attributes")
attribute_summary.add_column("Attribute", style="cyan")
attribute_summary.add_column("Kind", style="magenta")
attribute_summary.add_column("Unique")
for name in persona_dataset.attribute_names:
    info = persona_dataset.attribute_info(name)
    attribute_summary.add_row(
        name,
        info.get("kind", ""),
        str(info.get("n_unique_seed_values", "")),
    )
console.print(attribute_summary)

# %% Attribute-colored PCA views
# HACK: Just for current tsting speedup
# VARIANTS = ["biography"]
VARIANTS = ["biography"]

for variant, s in samples.items():
    build_layered_figure(
        s,
        "pca",
        title=f"PCA (2D) - {variant} - colored by age",
        n_components=2,
        **attribute_color_kwargs(persona_dataset, "age", persona_ids),
    ).show()

# %% Attribute-colored UMAP views
ATTRIBUTE = "total_wealth"
for variant, s in samples.items():
    build_layered_figure(
        s,
        "umap",
        title=f"UMAP (3D) - {variant} - colored by {ATTRIBUTE}",
        n_components=3,
        **attribute_color_kwargs(persona_dataset, ATTRIBUTE, persona_ids),
    ).show()

# %% Spectral-clustering-colored PCA views
# Spectral clustering on the kNN affinity graph recovers non-convex persona
# structure that k-means misses. Pick N_CLUSTERS from the eigengap plot above.
N_CLUSTERS = 3
for variant, s in samples.items():
    build_layered_figure(
        s,
        "pca",
        title=f"PCA (3D) - {variant} - spectral clusters",
        n_components=3,
        n_clusters=N_CLUSTERS,
        cluster_method="spectral",
    ).show()

# %% Attribute-colored Isomap views with kNN graph overlay
# for variant, s in samples.items():
#     build_layered_figure(
#         s,
#         "isomap",
#         title=f"Isomap (3D) - {variant} - colored by {ATTRIBUTE}",
#         n_components=3,
#         graph_overlay=True,
#         **attribute_color_kwargs(persona_dataset, ATTRIBUTE, persona_ids),
#     ).show()
