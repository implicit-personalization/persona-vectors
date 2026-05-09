#!/usr/bin/env python

"""Compare persona vectors from the Hub or local artifacts."""

# %% Imports

from itertools import combinations

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
    plot_layer_similarity,
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

# %% PCA - layered view per variant
for variant, s in samples.items():
    build_layered_figure(s, "pca", title=f"PCA - {variant} persona vectors").show()

# %% PCA (3D) - layered view per variant
for variant, s in samples.items():
    build_layered_figure(
        s,
        "pca",
        title=f"PCA (3D) - {variant} persona vectors",
        n_components=3,
    ).show()

# %% Centered similarity matrix - layered view per variant
MAX_PAIR_PERSONAS = 10
samples_small = {
    variant: load_persona_vectors(
        store, variant, persona_ids=persona_ids[:MAX_PAIR_PERSONAS]
    )
    for variant in comparison_variants
}

for variant, s in samples_small.items():
    build_layered_figure(
        s,
        "similarity",
        title=f"Centered similarity - {variant} persona vectors",
    ).show()

# %% Pair similarity - centered cosine trajectories per variant
for variant, s in samples_small.items():
    build_pair_similarity_figure(
        s,
        title=f"Pair similarity trajectories - {variant} persona vectors",
    ).show()

# %% Prompt-variant similarity - averaged across personas
avg_variant_vectors = {variant: s.vectors.mean(dim=0) for variant, s in samples.items()}
pair_traces = [
    (f"{left} vs {right}", avg_variant_vectors[left], avg_variant_vectors[right])
    for left, right in combinations(avg_variant_vectors, 2)
]

plot_layer_similarity(
    pair_traces,
    title="Layer-wise cosine similarity - averaged across personas",
    show=True,
)

# %% Prompt-variant similarity - one trace per persona
# This is the detailed original comparison view; it can get busy with many personas.
comparison_pairs = list(combinations(comparison_variants, 2))
persona_labels = next(iter(samples.values())).labels
all_pair_traces = []

for persona_index, persona_name in enumerate(persona_labels):
    all_pair_traces.extend(
        (
            f"{persona_name}: {left} vs {right}",
            samples[left].vectors[persona_index],
            samples[right].vectors[persona_index],
        )
        for left, right in comparison_pairs
    )

plot_layer_similarity(
    all_pair_traces,
    title="Layer-wise cosine similarity - all personas and variant pairs",
    show=True,
)
