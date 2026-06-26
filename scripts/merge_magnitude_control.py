#!/usr/bin/env python
"""Magnitude-vs-direction controls for the trait-merge correlation result.

The original merge generator was never committed; this reconstructs it on the
current trait-band API (``merge_trait_bands``) and adds the controls that settle
*why* co-steering correlated traits reinforces. Three steering runs on NDIF,
each writing an artifact consumed by ``build_trait_report.py`` §4:

* ``run_mcq_control``  -> merge_control.json          (solo / joint / magnitude-matched)
* ``run_dirnorm``      -> merge_control_dirnorm.json   (joint *direction* at solo magnitude)
* ``run_fluency``      -> merge_control_fluency.json   (repeat-frac of free text)

Headline: P(+pole) tracks total push magnitude in the correlated subspace,
*direction-agnostic* — steering one attribute harder (solo 3.29x) reproduces the
joint effect; the joint blend at solo magnitude does not. Co-steering's only edge
is fluency. Readout is MCQ option-probability (deterministic, no judge) under a
generic-human context (PERSONA_SYS), without which the bare model saturates each
attribute at its prior pole and steering has no room to move.
"""
import json
import os
import re
from pathlib import Path

import torch
from dotenv import load_dotenv
from nnsight import CONFIG
from nnterp import StandardizedTransformer
from persona_data.synth_persona import SynthPersonaDataset

from persona_vectors.artifacts import TraitVectorStore
from persona_vectors.extraction import MaskStrategy
from persona_vectors.steering import generate_band_steered
from persona_vectors.traits import merge_trait_bands

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "artifacts" / "trait_report"

MODEL_NAME = "google/gemma-2-9b-it"
MS = MaskStrategy.PERSONA_MEAN
BAND = list(range(14, 31))
# correlated set: born_in_us / citizenship / language (|cos| 0.03-0.29)
CORR = ["born_in_us", "us_citizenship_status", "speak_other_language"]
CZ = "us_citizenship_status"
# joint puts 1.76x citizenship's own-axis magnitude and 3.29x its total norm.
OWN_AXIS_MATCH = 1.76
NORM_MATCH = 3.29

PERSONA_SYS = (
    "You are a human being having a casual conversation. Stay in character and "
    "answer in the first person as a real person. Never say you are an AI."
)
MC_INSTRUCTION = (
    "Answer the multiple-choice question with your single best guess.\n"
    "Return exactly one uppercase letter."
)
LETTERS = ["A", "B", "C", "D", "E", "F"]


# ── helpers ───────────────────────────────────────────────────────────────────
def mcq_prompt(model, qa) -> tuple[str, list[int], int]:
    """(formatted prompt, option-letter token ids, +pole option index) for a seed MCQ.

    Drops the 'not enough information' escape (the model otherwise parks there with no
    persona context) and frames the question with PERSONA_SYS so the answer is steerable.
    """
    opts = [str(c).strip() for c in qa.choices if "not enough" not in str(c).lower()]
    letters = LETTERS[: len(opts)]
    body = "\n".join(f"{letters[i]}. {opts[i]}" for i in range(len(opts)))
    user = f"{PERSONA_SYS}\n\n{MC_INSTRUCTION}\n\nQUESTION:\n{qa.question.strip()}\n\nOPTIONS:\n{body}\n\nANSWER:"
    prompt = model.tokenizer.apply_chat_template(
        [{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True
    )
    letter_ids = [model.tokenizer(l, add_special_tokens=False).input_ids[0] for l in letters]
    return prompt, letter_ids, opts


def score_mcq(model, prompt, letter_ids, layer_vectors) -> list[float]:
    """P over option letters at the answer position under band steering (one remote trace)."""
    ids = model.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids
    pos = ids.shape[1] - 1
    for attempt in range(4):  # NDIF drops are intermittent
        try:
            with model.trace(ids, remote=True):
                for layer in sorted(layer_vectors):
                    model.steer(layers=layer, steering_vector=layer_vectors[layer], factor=1.0)
                lp = torch.log_softmax(model.logits[0, pos, letter_ids].float(), dim=-1).save()
            return lp.exp().detach().cpu().numpy().tolist()
        except Exception:
            if attempt == 3:
                raise


def renorm_per_layer(src: dict, target: dict) -> dict:
    """Scale each layer of ``src`` so its norm matches ``target`` at that layer."""
    return {L: src[L] * (target[L].norm() / (src[L].norm() + 1e-8)) for L in src}


def rep_frac(text: str) -> float:
    """Fraction of repeated word bigrams (0 = none, higher = degenerate)."""
    w = re.findall(r"\w+", text.lower())
    bg = list(zip(w, w[1:]))
    return round(1 - len(set(bg)) / len(bg), 3) if bg else 0.0


def seed_mcqs(dataset, attributes):
    """One bare seed MCQ per attribute (persona-independent)."""
    out = {}
    for pid in dataset.persona_ids:
        for q in dataset.get_qa(pid):
            if q.item_type == "mcq":
                for a in attributes:
                    if f"explicit_seed_attribute_{a}" in q.qid and a not in out:
                        out[a] = q
        if all(a in out for a in attributes):
            break
    return out


# ── experiments ───────────────────────────────────────────────────────────────
def run_mcq_control(model, store, qas, pole):
    """Solo / joint / magnitude-matched P(+pole) per correlated attribute."""
    joint = merge_trait_bands(store, CORR, BAND, strength=1.0, mask_strategy=MS)
    results = {}
    for a in CORR:
        prompt, letter_ids, opts = mcq_prompt(model, qas[a])
        to_idx = opts.index(pole[a])
        solo = merge_trait_bands(store, [a], BAND, strength=1.0, mask_strategy=MS)
        conds = {
            "unsteered": merge_trait_bands(store, [a], BAND, strength=0.0, mask_strategy=MS),
            "solo_t1": solo,
            "joint_t1": joint,
        }
        if a == CZ:  # the magnitude controls only on the attribute that barely steers alone
            conds[f"solo_matched_{OWN_AXIS_MATCH}"] = merge_trait_bands(
                store, [a], BAND, strength=OWN_AXIS_MATCH, mask_strategy=MS
            )
            conds[f"solo_normmatched_{NORM_MATCH}"] = merge_trait_bands(
                store, [a], BAND, strength=NORM_MATCH, mask_strategy=MS
            )
        p = {n: round(score_mcq(model, prompt, letter_ids, lv)[to_idx], 3) for n, lv in conds.items()}
        for n, v in p.items():
            print(f"  {a:24s} {n:22s} P(+pole)={v}")
        results[a] = {"pole": pole[a], "options": opts, "p_pole": p}
    (OUTDIR / "merge_control.json").write_text(json.dumps(results, indent=2))
    return results


def run_dirnorm(model, store, qas, pole):
    """Direction-at-fixed-magnitude: joint blend renormalised down to citizenship's solo norm."""
    prompt, letter_ids, opts = mcq_prompt(model, qas[CZ])
    to_idx = opts.index(pole[CZ])
    solo = merge_trait_bands(store, [CZ], BAND, strength=1.0, mask_strategy=MS)
    joint = merge_trait_bands(store, CORR, BAND, strength=1.0, mask_strategy=MS)
    conds = {
        "solo_t1": solo,
        "joint_t1": joint,
        "joint_dir_at_solo_norm": renorm_per_layer(joint, solo),
    }
    results = {}
    for n, lv in conds.items():
        mid = round(float(lv[BAND[len(BAND) // 2]].norm()), 2)
        p = round(score_mcq(model, prompt, letter_ids, lv)[to_idx], 3)
        results[n] = {"mid_layer_norm": mid, "p_pole": p}
        print(f"  {n:24s} mid|v|={mid:6.2f}  P(+pole)={p}")
    (OUTDIR / "merge_control_dirnorm.json").write_text(json.dumps(results, indent=2))
    return results


def run_fluency(model, store):
    """Free-text repeat-frac at matched total norm: joint (each t=1) vs solo citizenship 3.29x."""
    prompt = "Tell me about your background — where you're from and the languages you speak."
    conds = {
        "unsteered": merge_trait_bands(store, [CZ], BAND, strength=0.0, mask_strategy=MS),
        "joint_t1": merge_trait_bands(store, CORR, BAND, strength=1.0, mask_strategy=MS),
        f"solo_normmatched_{NORM_MATCH}": merge_trait_bands(
            store, [CZ], BAND, strength=NORM_MATCH, mask_strategy=MS
        ),
    }
    results = {}
    for n, lv in conds.items():
        text = generate_band_steered(
            model, prompt, lv, system=PERSONA_SYS, max_new_tokens=120, remote=True
        )
        results[n] = {"rep_frac": rep_frac(text), "text": text}
        print(f"\n==== {n}  rep_frac={results[n]['rep_frac']} ====\n{text}")
    (OUTDIR / "merge_control_fluency.json").write_text(json.dumps(results, indent=2))
    return results


def main():
    load_dotenv(ROOT / ".env")
    CONFIG.API.APIKEY = os.environ["NDIF_API_KEY"]
    torch.set_grad_enabled(False)
    OUTDIR.mkdir(parents=True, exist_ok=True)

    dataset = SynthPersonaDataset()
    store = TraitVectorStore(MODEL_NAME)
    qas = seed_mcqs(dataset, CORR)
    pole = {a: store.metadata(a, mask_strategy=MS)["value_to"] for a in CORR}

    model = StandardizedTransformer(MODEL_NAME, check_attn_probs_with_trace=False)

    print("\n# MCQ magnitude control")
    run_mcq_control(model, store, qas, pole)
    print("\n# Direction at fixed magnitude")
    run_dirnorm(model, store, qas, pole)
    print("\n# Fluency at matched total norm")
    run_fluency(model, store)
    print(f"\nwrote artifacts to {OUTDIR}")


if __name__ == "__main__":
    main()
