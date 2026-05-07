#!/usr/bin/env python

"""Compare persona vectors loaded directly from a Hugging Face dataset."""

import torch
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from persona_vectors.analysis import (
    list_comparison_personas,
    load_persona_vectors,
    pca_explained_variance,
)
from persona_vectors.artifacts import HFActivationStore
from persona_vectors.extraction import MaskStrategy
from persona_vectors.plots import (
    build_layered_figure,
    build_pair_similarity_figure,
    plot_scree,
)

console = Console()

# %% Setup
load_dotenv()
torch.set_grad_enabled(False)

REPO_ID = "implicit-personalization/synth-persona-vectors"
MODEL_NAME = "google/gemma-2-9b-it"
MASK_STRATEGY = MaskStrategy.ANSWER_MEAN

# Biography may not be available in the published dataset yet, so prefer it
# when present and otherwise fall back to the first available split.
PREFERRED_VARIANTS = ["biography", "templated"]

# %% Load Hub activation store
store = HFActivationStore(REPO_ID, MODEL_NAME, mask_strategy=MASK_STRATEGY)
available_variants = store.available_variants(PREFERRED_VARIANTS)

variant = available_variants[0]
persona_ids = list_comparison_personas(store, [variant])

summary = Table(title="Hub Activation Dataset")
summary.add_column("Property", style="cyan")
summary.add_column("Value", style="magenta")
summary.add_row("Repo", REPO_ID)
summary.add_row("Config", store.config_name)
summary.add_row("Available variants", ", ".join(available_variants))
summary.add_row("Selected variant", variant)
summary.add_row("Personas loaded", str(len(persona_ids)))
console.print(summary)

# %% Load persona vectors directly from the Hub dataset
samples = load_persona_vectors(store, variant, persona_ids=persona_ids)

# %% Scree plot of PCA explained variance for a few representative layers
# NOTE: Probably looking at this the first 5-6 components are the most important ones (encoding most of the info)
num_layers = int(samples.vectors.shape[1])
scree_layers = sorted({0, num_layers // 3, (2 * num_layers) // 3, num_layers - 1})
plot_scree(
    {
        f"layer {layer}": pca_explained_variance(samples.vectors[:, layer, :])
        for layer in scree_layers
    },
    title=f"Hub PCA explained variance - {variant} persona vectors",
    show=True,
)

# %% Simple PCA between personas
build_layered_figure(
    samples, "pca", title=f"Hub PCA - {variant} persona vectors"
).show()

# %% Optional centered cosine view over the same Hub-loaded vectors
MAX_PERSONAS = 10
samples = load_persona_vectors(store, variant, persona_ids=persona_ids[:MAX_PERSONAS])
build_pair_similarity_figure(
    samples, title=f"Hub pair similarity - {variant} persona vectors"
).show()
