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
VARIANTS = ["biography", "templated"]
INCLUDE_BASELINE = False  # set True to include the baseline_assistant persona

# %% Load Hub activation store
store = HFActivationStore(REPO_ID, MODEL_NAME, mask_strategy=MASK_STRATEGY)
available_variants = store.available_variants(VARIANTS)

comparison_variants = [variant for variant in VARIANTS if variant in available_variants]
persona_ids = list_comparison_personas(
    store, comparison_variants, include_baseline=INCLUDE_BASELINE
)

summary = Table(title="Hub Activation Dataset")
summary.add_column("Property", style="cyan")
summary.add_column("Value", style="magenta")
summary.add_row("Repo", REPO_ID)
summary.add_row("Config", store.config_name)
summary.add_row("Available variants", ", ".join(available_variants))
summary.add_row("Compared variants", ", ".join(comparison_variants))
summary.add_row("Personas loaded", str(len(persona_ids)))
console.print(summary)

# %% Load persona vectors for each variant
samples = {
    variant: load_persona_vectors(store, variant, persona_ids=persona_ids)
    for variant in comparison_variants
}

# %% Scree plot — PCA explained variance for representative layers
# NOTE: Probably looking at this the first 5-6 components are the most important ones (encoding most of the info)
for variant, s in samples.items():
    num_layers = int(s.vectors.shape[1])
    scree_layers = sorted({0, num_layers // 3, (2 * num_layers) // 3, num_layers - 1})
    plot_scree(
        {
            f"layer {layer}": pca_explained_variance(s.vectors[:, layer, :])
            for layer in scree_layers
        },
        title=f"Hub PCA explained variance - {variant} persona vectors",
        show=True,
    )

# %% PCA — layered view per variant
for variant, s in samples.items():
    build_layered_figure(s, "pca", title=f"Hub PCA - {variant} persona vectors").show()

# %% Pair similarity — centered cosine view per variant
MAX_PERSONAS = 10
for variant in comparison_variants:
    s = load_persona_vectors(store, variant, persona_ids=persona_ids[:MAX_PERSONAS])
    build_pair_similarity_figure(
        s, title=f"Hub pair similarity - {variant} persona vectors"
    ).show()
