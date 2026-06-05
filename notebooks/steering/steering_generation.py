#!/usr/bin/env python

# # Watching generation change under persona steering
#
# Build one signed difference-of-means axis for a persona attribute, add it to
# the residual stream during generation on a live NDIF model, and read off how
# the text changes. Two views:
#
#   1. **Free-form** — a human-persona prompt swept - / 0 / + .
#   2. **In-context override** — a conflicting persona in the system prompt that we try to overpower by steering toward the opposite class.
#
# Swap `ATTR` / `ALPHAS` and re-run. What makes act-add steering actually work:
#
#   1. gap units: `coefficient = alpha * gap_norm`, alpha=1 = opposite-class
#      centroid. Scaling by the full residual norm over-steers 8-28x the real class gap and collapses the model into degenerate text
#   2. layer: AUC (decodability) is ~flat across layers, so the argmax-AUC layer is a noisy
#      pick and usually too late to steer generation. Steer a mid layer (`STEER_LAYER`); the
#      layer-sweep cell shows which layers actually move the stated age.
#   3. scenario: the prompt must let the model *choose* the age. A neutral prompt makes
#      gemma-it refuse ("I'm an AI"); a templated persona pins the age as a literal token the
#      copy/induction circuit just reproduces (section 2 — steering can't override it at any
#      layer/strength). `PERSONA_SYS` frames a human without stating an age, so the steered
#      prior is free to set it. The diff-of-means axis is causally correct as-is
#      (+coefficient => `positive`, e.g. older), so no sign flip is needed — `SIGN_CAL` stays empty.

# %% Setup
import torch
from dotenv import load_dotenv
from nnterp import StandardizedTransformer
from persona_data.prompts import format_prompt
from persona_data.synth_persona import SynthPersonaDataset

from persona_vectors.artifacts import HFPersonaVectorStore
from persona_vectors.extraction import MaskStrategy
from persona_vectors.probes import attribute_probe_labels
from persona_vectors.steer_generate import (
    _contrast_labels,
    build_attribute_direction,
    generate_steered,
    steering_coefficient,
)

REPO_ID = "implicit-personalization/synth-persona-vectors"
MODEL_NAME = "google/gemma-2-9b-it"  # NOTE: Model is hot on ndif and we have persona-vectors for it

PROMPT = (
    "Introduce yourself: how old are you, and what is a typical day in your life like?"
)

ATTR = "age"  # swap to any attribute

# signed strength in gap units; baseline 0 added below
ALPHAS = [-6.0, -3.0, 3.0, 6.0]

# mid-stack: AUC is ~flat across layers, so the best *causal* layer is not the argmax-AUC one the selector would pick
STEER_LAYER = 20

# Human framing so an age is actually stated. A neutral prompt makes gemma-it refuse
# ("I'm an AI"); a templated persona pins the age as a literal token it just copies. This
# lets the steered prior choose the number (contrast with the in-context override below).
PERSONA_SYS = (
    "You are a human being having a casual conversation. Stay in character and "
    "answer in the first person as a real person with a real life. Never say you are an AI."
)

# Per-attribute causal-sign flip so +alpha == the printed `positive` label. The
# difference-of-means axis (positive-minus-negative) is causally correct as-is at these
# layers: +coefficient steers toward `positive` (age: older), matching steering_mcq.py.
SIGN_CAL: dict[str, float] = {}

load_dotenv()
torch.set_grad_enabled(False)

# %% Load store, dataset, and the live model Build the model LOCALLY (config/meta only);
store = HFPersonaVectorStore(
    REPO_ID, MODEL_NAME, mask_strategy=MaskStrategy.ANSWER_MEAN
)

ids = store.list_personas(["templated"], include_baseline=False)
n_layers = int(store.load("templated", ids[0]).shape[0])

candidate_layers = list(range(8, n_layers, max(3, n_layers // 12)))
dataset = SynthPersonaDataset()
model = StandardizedTransformer(MODEL_NAME)
print(f"{MODEL_NAME}: {len(ids)} personas, {n_layers} layers")

# %% Build the steering axis Difference of means between the two groups (median split for numeric/ordinal, one-vs-rest for categorical), at STEER_LAYER. `+` means the printed `positive` label.
info = build_attribute_direction(
    store,
    dataset,
    ATTR,
    variant="templated",
    candidate_layers=[STEER_LAYER],
    persona_ids=ids,
)
sign = SIGN_CAL.get(ATTR, 1.0)  # causal-sign flip so +alpha == info['positive']
print(
    f"{ATTR}: + = {info['positive']!r}  layer={info['layer']}  AUC={info['auc']:.3f}  "
    f"gap_norm={info['gap_norm']:.1f}  sign_cal={sign:+g}"
)


# %% 1. Free-form: human-persona prompt, swept - / 0 / +

# Same prompt every time; only the steered axis changes, so any difference is the direction's
# causal effect. PERSONA_SYS frames a human without pinning an age, so steering can move it.
coeffs = [steering_coefficient(info, a, sign=sign) for a in ALPHAS]
out = generate_steered(
    model,
    PROMPT,
    info["layer"],
    info["unit_direction"],
    [0.0] + coeffs,
    system=PERSONA_SYS,
    max_new_tokens=200,  # min_new_tokens is pinned to this in generate_steered
    remote=True,
)
for a, c in zip([0.0] + ALPHAS, [0.0] + coeffs):
    print(f"\n==== alpha={a:+g} ====\n{out[float(c)]}")

# %% Layer sweep: which layer actually steers the stated age?
#
# AUC (decodability) is ~flat across layers, so it can't tell us where steering is causally
# strongest. Rebuild the axis at each candidate layer, steer one fixed alpha, and read off the
# stated age. This is how STEER_LAYER above was chosen — pick a layer that moves the number
# without breaking coherence.
SWEEP_ALPHA = 6.0
for L in candidate_layers:
    li = build_attribute_direction(
        store,
        dataset,
        ATTR,
        variant="templated",
        candidate_layers=[L],
        persona_ids=ids,
    )
    txt = generate_steered(
        model,
        PROMPT,
        li["layer"],
        li["unit_direction"],
        [steering_coefficient(li, SWEEP_ALPHA, sign=sign)],
        system=PERSONA_SYS,
        max_new_tokens=48,
        remote=True,
    )
    print(f"\n==== layer={L} (alpha={SWEEP_ALPHA:+g}) ====\n{next(iter(txt.values()))}")

# %% 2. In-context override

# Put a persona from the *negative* group in the system prompt, then steer toward the positive
# class. The age is now a literal token in context, so the copy/induction circuit reproduces it
# verbatim and steering can't move the number at any layer/strength — only the prose style shifts.
y01, keep, positive = _contrast_labels(attribute_probe_labels(dataset, ATTR, ids))
neg_id = next(pid for pid, y, k in zip(ids, y01, keep) if k and y == 0)
sys_prompt = format_prompt(
    dataset._personas_by_id[neg_id], "templated", mode="conversational"
)
print(f"in-context persona is in the negative group; steering + toward {positive!r}")

ic_alphas = [0.0, 1.0, 1.5, 2.0]
ic = generate_steered(
    model,
    PROMPT,
    info["layer"],
    info["unit_direction"],
    [steering_coefficient(info, a, sign=sign) for a in ic_alphas],
    system=sys_prompt,
    max_new_tokens=200,
    remote=True,
)

for a in ic_alphas:
    print(f"\n==== alpha={a:+g} ====\n{ic[steering_coefficient(info, a, sign=sign)]}")
