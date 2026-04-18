#!/usr/bin/env python

"""Layer-wise cosine similarity between prompt variants.

Compares all pairs of available variants (templated, biography, baseline)
for one or more personas. Each persona gets its own set of traces.

Research notes:
- Centering activations per layer along the feature dimension can change
  the cosine similarity picture. See "Mech interp puzzle 1: Suspiciously
  similar embeddings in GPT" for background.
  https://www.alignmentforum.org/posts/eLNo7b56kQQerCzp2/
  TODO: revisit centering approach; current comparison is uncentered.
"""

from itertools import combinations

import torch
from dotenv import load_dotenv
from persona_data.synth_persona import SynthPersonaDataset
from rich.console import Console
from rich.table import Table

from persona_vectors.analysis import pairwise_cosine_similarity
from persona_vectors.artifacts import (
    SUPPORTED_VARIANTS,
    ActivationStore,
    list_personas,
    load_persona_names,
)
from persona_vectors.plots import plot_layer_similarity, plot_similarity_matrix_grid

console = Console()

# %% Setup
load_dotenv()
torch.set_grad_enabled(False)

# Use 9b for remote (production), 2b for local testing
REMOTE = False
# REMOTE = True
MODEL_NAME = "google/gemma-2-9b-it" if REMOTE else "google/gemma-2-2b-it"

# %% Load dataset and Activations
dataset = SynthPersonaDataset()
acts = ActivationStore(MODEL_NAME)

dataset_table = Table(title="Dataset")
dataset_table.add_column("Property", style="cyan")
dataset_table.add_column("Value", style="magenta")
dataset_table.add_row("Total Personas", str(len(dataset)))
dataset_table.add_row("First Persona", dataset[0].name)
dataset_table.add_row("Model Name", acts.model_name)
console.print(dataset_table)

# %% Discover which variants are available
available_variants = [
    variant
    for variant in SUPPORTED_VARIANTS
    if list_personas(acts.root_dir, MODEL_NAME, [variant])
]
console.print(f"Available variants: {available_variants}")

persona_ids = list_personas(acts.root_dir, MODEL_NAME, available_variants)
persona_names = load_persona_names(
    acts.root_dir, MODEL_NAME, available_variants, persona_ids
)
console.print(f"Personas with all variants: {len(persona_ids)}")

# %% Load mean activations per variant per persona
variant_means: dict[str, dict[str, torch.Tensor]] = {}
for variant in available_variants:
    variant_means[variant] = {}
    for pid in persona_ids:
        activations, _ = acts.load(variant, pid)
        variant_means[variant][pid] = activations.float().mean(dim=0)

# %% Plot all variant pairs for each persona
comparison_pairs = list(combinations(available_variants, 2))

for pid in persona_ids:
    persona_name = persona_names.get(pid, pid[:8])
    pair_traces = [
        (f"{left} vs {right}", variant_means[left][pid], variant_means[right][pid])
        for left, right in comparison_pairs
    ]

    plot_layer_similarity(
        pair_traces,
        title=f"Layer-wise Cosine Similarity — {persona_name}",
        # NOTE: This adds a lot of plot creations so currently disabled
        show=False,
    )

# %% Plot Averaged across personas
avg_variant_means = {
    variant: torch.stack([variant_means[variant][pid] for pid in persona_ids]).mean(
        dim=0
    )
    for variant in available_variants
}
pair_traces = [
    (f"{left} vs {right}", avg_variant_means[left], avg_variant_means[right])
    for left, right in comparison_pairs
]

plot_layer_similarity(
    pair_traces,
    title="Layer-wise Cosine Similarity — Averaged across personas",
    show=True,
)

# TODO: Clean up removing the average or something else this is too noisy
# Not really resonable to have those results with such an high similarity
# %% Plot pairwise similarity matrices at 4 layers
# matrix_variant = available_variants[0]
# layer_count = variant_means[matrix_variant][persona_ids[0]].shape[0]
# layer_indices = [round(i * (layer_count - 1) / 3) for i in range(4)]
# matrix_titles = [f"Layer {layer_index + 1}" for layer_index in layer_indices]
# labels = [persona_names.get(pid, pid[:8]) for pid in persona_ids]
#
# similarity_matrices = [
#     pairwise_cosine_similarity(
#         [variant_means[matrix_variant][pid][layer_index] for pid in persona_ids]
#     )
#     for layer_index in layer_indices
# ]
#
# plot_similarity_matrix_grid(
#     similarity_matrices,
#     labels=labels,
#     titles=matrix_titles,
#     title=f"Pairwise Cosine Similarity Across Personas ({matrix_variant})",
#     show=True,
# )
