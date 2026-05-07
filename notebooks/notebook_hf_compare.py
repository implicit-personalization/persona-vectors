#!/usr/bin/env python

"""Compare persona vectors loaded directly from a Hugging Face dataset."""

import torch
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from persona_vectors.analysis import list_comparison_personas, load_persona_vectors
from persona_vectors.artifacts import HFActivationStore
from persona_vectors.extraction import MaskStrategy
from persona_vectors.plots import build_layered_figure, build_pair_similarity_figure

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
if not available_variants:
    raise SystemExit(f"No Hub variants found for {store.config_name}")

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
