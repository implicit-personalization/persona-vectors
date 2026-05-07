#!/usr/bin/env python

"""Layer-wise cosine similarity between prompt variants."""

from itertools import combinations

import torch
from dotenv import load_dotenv
from persona_data.synth_persona import SynthPersonaDataset
from rich.console import Console
from rich.table import Table

from persona_vectors.analysis import (
    list_comparison_personas,
    load_persona_vectors,
    load_variant_vectors,
)
from persona_vectors.artifacts import ActivationStore
from persona_vectors.extraction import MaskStrategy
from persona_vectors.plots import (
    build_layered_figure,
    build_pair_similarity_figure,
    plot_layer_similarity,
)

console = Console()

# %% Setup
load_dotenv()
torch.set_grad_enabled(False)

# Use 9b for remote (production), 2b for local testing
# REMOTE = False
REMOTE = True
MODEL_NAME = "google/gemma-2-9b-it" if REMOTE else "google/gemma-2-2b-it"
MASK_STRATEGY = MaskStrategy.ANSWER_MEAN
SIMILARITY_VARIANT = "biography"
INCLUDE_BASELINE = False

# %% Load dataset and Activations
dataset = SynthPersonaDataset()
acts = ActivationStore(MODEL_NAME, mask_strategy=MASK_STRATEGY)

dataset_table = Table(title="Dataset")
dataset_table.add_column("Property", style="cyan")
dataset_table.add_column("Value", style="magenta")
dataset_table.add_row("Total Personas", str(len(dataset)))
dataset_table.add_row("First Persona", dataset[0].name)
dataset_table.add_row("Model Name", acts.model_name)
console.print(dataset_table)

# %% Discover which variants are available
available_variants = acts.available_variants()
console.print(f"Available comparison variants: {available_variants}")

persona_ids = list_comparison_personas(
    acts,
    available_variants,
    include_baseline=INCLUDE_BASELINE,
)
console.print(f"Personas with all variants: {len(persona_ids)}")

# %% Load persona vectors per variant
variant_samples = load_variant_vectors(
    acts, available_variants, persona_ids=persona_ids
)

persona_labels = next(iter(variant_samples.values())).labels

# %% Plot all persona/variant-pair traces together
comparison_pairs = list(combinations(available_variants, 2))
all_pair_traces = []

for persona_index, persona_name in enumerate(persona_labels):
    all_pair_traces.extend(
        (
            f"{persona_name}: {left} vs {right}",
            variant_samples[left].vectors[persona_index],
            variant_samples[right].vectors[persona_index],
        )
        for left, right in comparison_pairs
    )

plot_layer_similarity(
    all_pair_traces,
    title="Layer-wise Cosine Similarity — All personas and variant pairs",
    show=True,
)

# %% Plot Averaged across personas
avg_variant_vectors = {
    variant: samples.vectors.mean(dim=0) for variant, samples in variant_samples.items()
}

pair_traces = [
    (f"{left} vs {right}", avg_variant_vectors[left], avg_variant_vectors[right])
    for left, right in combinations(avg_variant_vectors, 2)
]

plot_layer_similarity(
    pair_traces,
    title=("Layer-wise Cosine Similarity — Averaged across personas"),
    show=True,
)

# %% Similarity matrix and pair trajectories, matching the UI comparison view
similarity_samples = load_persona_vectors(
    acts, SIMILARITY_VARIANT, persona_ids=persona_ids
)

build_layered_figure(
    similarity_samples,
    "similarity",
    title=f"Centered similarity — {SIMILARITY_VARIANT} — persona vectors",
).show()

build_pair_similarity_figure(
    similarity_samples,
    title=(f"Pair similarity trajectories — {SIMILARITY_VARIANT} — persona vectors"),
).show()
