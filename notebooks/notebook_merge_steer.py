#!/usr/bin/env python

# # Merging trait vectors and steering with the result
#
# `notebook_extract_trait.py` builds one deconfounded vector per binary attribute
# and steers with a single one. Because trait vectors share one residual space,
# several can be **merged** (summed per layer) and applied together — this is the
# `merge_trait_bands` helper. Here we co-steer a *correlated* set
# (born_in_us · citizenship · language) and compare it against steering each
# attribute solo, on a neutral human prompt.
#
# Run `notebook_extract_trait.py` first so the trait vectors exist in
# artifacts/trait_vectors/<model>/. The rigorous magnitude-vs-direction control
# behind this lives in `scripts/merge_magnitude_control.py` (report §4).

# %% Setup
import torch
from dotenv import load_dotenv
from nnterp import StandardizedTransformer
from rich.console import Console

from persona_vectors.artifacts import TraitVectorStore
from persona_vectors.extraction import MaskStrategy
from persona_vectors.steering import generate_band_steered
from persona_vectors.traits import merge_trait_bands

console = Console()
load_dotenv()
torch.set_grad_enabled(False)

# %% Model + config
# Use 9b for remote (its trait vectors ship in artifacts/); 2b for local testing
# requires re-running notebook_extract_trait.py at 2b first.
REMOTE = True
MODEL_NAME = "google/gemma-2-9b-it" if REMOTE else "google/gemma-2-2b-it"
MASK_STRATEGY = MaskStrategy.PERSONA_MEAN

print(f"Loading {MODEL_NAME}...")
model = StandardizedTransformer(MODEL_NAME, check_attn_probs_with_trace=False)
mid = model.num_layers // 2
BAND = list(range(mid - 7, mid + 10))  # mid-stack band, e.g. 14–30 on gemma-2-9b

# Correlated set: each carries an "American" pole; merged they should compound into
# a coherent non-US, non-citizen, multilingual persona.
ATTRS = ["born_in_us", "us_citizenship_status", "speak_other_language"]

PERSONA_SYS = (
    "You are a human being having a casual conversation. Stay in character and "
    "answer in the first person as a real person. Never say you are an AI."
)
PROMPT = "Tell me about your background — where you're from and the languages you speak."

# %% Baseline (no steering)
store = TraitVectorStore(MODEL_NAME, mask_strategy=MASK_STRATEGY)
baseline = generate_band_steered(
    model,
    PROMPT,
    merge_trait_bands(store, ATTRS, BAND, strength=0.0, mask_strategy=MASK_STRATEGY),
    system=PERSONA_SYS,
    max_new_tokens=120,
    remote=REMOTE,
)
console.rule("baseline (unsteered)")
print(baseline)

# %% Each trait solo (one attribute's band at strength 1)
for attr in ATTRS:
    solo = merge_trait_bands(store, [attr], BAND, strength=1.0, mask_strategy=MASK_STRATEGY)
    text = generate_band_steered(
        model, PROMPT, solo, system=PERSONA_SYS, max_new_tokens=120, remote=REMOTE
    )
    console.rule(f"solo: {attr}")
    print(text)

# %% Merged (all three trait bands summed, each at strength 1)
joint = merge_trait_bands(store, ATTRS, BAND, strength=1.0, mask_strategy=MASK_STRATEGY)
merged = generate_band_steered(
    model, PROMPT, joint, system=PERSONA_SYS, max_new_tokens=120, remote=REMOTE
)
console.rule(f"merged: {' + '.join(ATTRS)}")
print(merged)
