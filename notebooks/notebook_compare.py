#!/usr/bin/env python

"""Layer-wise cosine similarity between prompt variants."""

from itertools import combinations

import torch
from dotenv import load_dotenv
from persona_data.prompts import BASELINE_PERSONA_ID
from persona_data.synth_persona import SynthPersonaDataset
from rich.console import Console
from rich.table import Table

from persona_vectors.analysis import load_persona_mean_samples
from persona_vectors.artifacts import (
    PERSONA_VARIANTS,
    ActivationStore,
    list_personas,
    load_persona_names,
)
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
INCLUDE_BASELINE_REFERENCE = True
SIMILARITY_VARIANT = "biography"
LAYERS: list[int] | None = None

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
    for variant in PERSONA_VARIANTS
    if list_personas(acts.root_dir, MODEL_NAME, [variant], mask_strategy=MASK_STRATEGY)
]
console.print(f"Available comparison variants: {available_variants}")
baseline_available = BASELINE_PERSONA_ID in list_personas(
    acts.root_dir,
    MODEL_NAME,
    [BASELINE_PERSONA_ID],
    mask_strategy=MASK_STRATEGY,
    warn_missing=False,
)
include_baseline = INCLUDE_BASELINE_REFERENCE and baseline_available
console.print(f"Baseline reference available: {baseline_available}")

persona_ids = list_personas(
    acts.root_dir, MODEL_NAME, available_variants, mask_strategy=MASK_STRATEGY
)
persona_names = load_persona_names(
    acts.root_dir,
    MODEL_NAME,
    available_variants,
    persona_ids,
    mask_strategy=MASK_STRATEGY,
)
console.print(f"Personas with all variants: {len(persona_ids)}")

# %% Load mean activations per variant per persona
variant_means: dict[str, dict[str, torch.Tensor]] = {}
for variant in available_variants:
    variant_means[variant] = {}
    for pid in persona_ids:
        activations, _ = acts.load(variant, pid, mask_strategy=MASK_STRATEGY)
        variant_means[variant][pid] = activations.float().mean(dim=0)

# %% Plot all persona/variant-pair traces together
comparison_pairs = list(combinations(available_variants, 2))
all_pair_traces = []

for pid in persona_ids:
    persona_name = persona_names.get(pid, pid[:8])
    all_pair_traces.extend(
        (
            f"{persona_name}: {left} vs {right}",
            variant_means[left][pid],
            variant_means[right][pid],
        )
        for left, right in comparison_pairs
    )

plot_layer_similarity(
    all_pair_traces,
    title="Layer-wise Cosine Similarity — All personas and variant pairs",
    show=True,
)

# %% Plot Averaged across personas
avg_variant_means = {
    variant: torch.stack([variant_means[variant][pid] for pid in persona_ids]).mean(
        dim=0
    )
    for variant in available_variants
}
avg_plot_means = dict(avg_variant_means)
if include_baseline:
    baseline_vectors, _ = acts.load(
        BASELINE_PERSONA_ID,
        BASELINE_PERSONA_ID,
        mask_strategy=MASK_STRATEGY,
    )
    avg_plot_means[BASELINE_PERSONA_ID] = baseline_vectors.float().mean(dim=0)

pair_traces = [
    (f"{left} vs {right}", avg_plot_means[left], avg_plot_means[right])
    for left, right in combinations(avg_plot_means, 2)
]

plot_layer_similarity(
    pair_traces,
    title=(
        "Layer-wise Cosine Similarity — Averaged across personas"
        + (" + baseline" if include_baseline else "")
    ),
    show=True,
)

# %% Similarity matrix and pair trajectories, matching the UI comparison view
similarity_variant = (
    SIMILARITY_VARIANT
    if SIMILARITY_VARIANT in available_variants
    else available_variants[0]
)
samples = load_persona_mean_samples(
    acts.root_dir,
    MODEL_NAME,
    similarity_variant,
    mask_strategy=MASK_STRATEGY,
    persona_ids=persona_ids,
    include_baseline=include_baseline,
)

build_layered_figure(
    samples,
    "similarity",
    layers=LAYERS,
    title=(
        "Centered similarity — "
        f"{similarity_variant} — personas averaged over questions"
        + (" + baseline" if include_baseline else "")
    ),
).show()

build_pair_similarity_figure(
    samples,
    layers=LAYERS,
    title=(
        "Pair similarity trajectories — "
        f"{similarity_variant} — personas averaged over questions"
        + (" + baseline" if include_baseline else "")
    ),
).show()
