#!/usr/bin/env python

"""Manifold and attribute views over persona vectors from Hub or local artifacts.

For pairwise similarity, dendrograms, and prompt-variant comparisons see
``similarity.py``.
"""

# %% Imports

import torch
from dotenv import load_dotenv
from persona_data.synth_persona import SynthPersonaDataset
from rich.console import Console
from rich.table import Table

from persona_vectors.analysis import load_persona_vectors, pca_explained_variance
from persona_vectors.artifacts import HFPersonaVectorStore
from persona_vectors.attributes import attribute_color_kwargs
from persona_vectors.extraction import MaskStrategy
from persona_vectors.plots import build_layered_figure, plot_scree

console = Console()

# %% Setup
load_dotenv()
torch.set_grad_enabled(False)

REPO_ID = "implicit-personalization/synth-persona-vectors"
# MODEL_NAME = "google/gemma-3-27b-it"
MODEL_NAME = "meta-llama/Llama-3.1-405B-Instruct"
MASK_STRATEGY = MaskStrategy.ANSWER_MEAN
VARIANT = "templated"  # the single prompt variant every view below uses
INCLUDE_BASELINE = False

# %% Load activation store
# Default: read the published Hub artifact dataset. To use local artifacts
# instead, comment out HFPersonaVectorStore and uncomment the two local lines.
store = HFPersonaVectorStore(REPO_ID, MODEL_NAME, mask_strategy=MASK_STRATEGY)

# NOTE: This is an example of how you may get things from another position locally

# store = PersonaVectorStore(
#     MODEL_NAME, mask_strategy=MASK_STRATEGY, root_dir="artifacts/persona-vectors"
# )

persona_ids = store.list_personas([VARIANT], include_baseline=INCLUDE_BASELINE)

summary = Table(title="Activation Dataset")
summary.add_column("Property", style="cyan")
summary.add_column("Value", style="magenta")
summary.add_row("Store", type(store).__name__)
summary.add_row("Repo", getattr(store, "repo_id", "local artifacts"))
summary.add_row("Model", store.model_name)
summary.add_row("Config", getattr(store, "config_name", str(MASK_STRATEGY)))
summary.add_row("Variant", VARIANT)
summary.add_row("Personas loaded", str(len(persona_ids)))
console.print(summary)

# %% Load persona vectors
s = load_persona_vectors(store, VARIANT, persona_ids=persona_ids)

# %% Scree plot - PCA explained variance for representative layers
# NOTE: Usually the first 5 components carry most of the visible structure.
num_layers = int(s.vectors.shape[1])
scree_layers = sorted({0, num_layers // 3, (2 * num_layers) // 3, num_layers - 1})
plot_scree(
    {
        f"layer {layer}": pca_explained_variance(s.vectors[:, layer, :])
        for layer in scree_layers
    },
    title=f"PCA explained variance - {VARIANT} persona vectors",
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
ATTRIBUTE = "age"
build_layered_figure(
    s,
    "pca",
    title=f"PCA (2D) - {VARIANT} - colored by {ATTRIBUTE}",
    n_components=2,
    **attribute_color_kwargs(persona_dataset, ATTRIBUTE, persona_ids),
).show()

# %% Attribute-colored UMAP views
ATTRIBUTE = "total_wealth"
build_layered_figure(
    s,
    "umap",
    title=f"UMAP (3D) - {VARIANT} - colored by {ATTRIBUTE}",
    n_components=3,
    **attribute_color_kwargs(persona_dataset, ATTRIBUTE, persona_ids),
).show()

# %% Attribute-colored Isomap views with kNN graph overlay
# build_layered_figure(
#     s,
#     "isomap",
#     title=f"Isomap (3D) - {VARIANT} - colored by {ATTRIBUTE}",
#     n_components=3,
#     graph_overlay=True,
#     **attribute_color_kwargs(persona_dataset, ATTRIBUTE, persona_ids),
# ).show()
