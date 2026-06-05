#!/usr/bin/env python
# # Does attribute steering move multiple-choice answers?
#
# Score the bare `explicit_seed_attribute` MCQs (e.g. "How old are you?", no persona context)
# while injecting an attribute axis at the answer position.
# We collapse the choice distribution to one scalar (prob-weighted attribute value) and ask whether it shifts monotonically with steering strength.
#
# Same act-add difference-of-means axis as `steering_generation.py` one vector, one layer.

# %% Setup
import json
import re
import time

import numpy as np
import torch
from dotenv import load_dotenv
from nnterp import StandardizedTransformer
from persona_data.environment import get_artifacts_dir
from persona_data.synth_persona import SynthPersonaDataset

from persona_vectors.artifacts import HFPersonaVectorStore
from persona_vectors.extraction import MaskStrategy
from persona_vectors.steer_generate import (
    build_attribute_direction,
    steering_coefficient,
)

REPO_ID = "implicit-personalization/synth-persona-vectors"
MODEL_NAME = "google/gemma-2-9b-it"

OUT = get_artifacts_dir() / "steering_mcq"

N_PERSONAS = 3
ATTRS = ["age", "total_wealth", "family_income_at_16", "highest_degree_received"]

# Signed strength in *gap units* (factor = t * gap_norm): t=1 lands exactly at the
# opposite-class centroid (one full inter-class gap), 0 = baseline. Scaling by the
# full residual norm instead over-steers 8-28x the real class separation
# (gap_norm is only 3.5-12% of act_norm) and collapses the model off-manifold. Coherent window is t in {0.5..3}.
TS = [-3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0]

# Forced choice over substantive options only — we drop the "Not enough information" escape, on which the model otherwise parks
# 100% (correct but uninformative) with no persona context.

# TODO: switch this to use persona-data prompt formatting to answer multiple choice questions
MC_INSTRUCTION = (
    "Answer the multiple-choice question with your single best guess.\n"
    "Return exactly one uppercase letter."
)

# Specific-before-general order matters: "above average" contains "average".
_INCOME_SCALE = [
    ("far below", 0),
    ("below average", 1),
    ("above average", 3),
    ("average", 2),
]
_DEGREE_SCALE = [
    ("less than high school", 0),
    ("high school", 1),
    ("associate", 2),
    ("junior college", 2),
    ("bachelor", 3),
    ("graduate", 4),
]


def choice_value(attribute, text):
    """Sortable scalar for a substantive choice; None for the 'not enough info' option."""
    t = text.strip().lower()
    if "not enough" in t or "sufficient" in t:
        return None
    if attribute == "age":
        m = re.search(r"\d+", t)
        return float(m.group()) if m else None
    if "income" in attribute:
        return next((float(v) for kw, v in _INCOME_SCALE if kw in t), None)
    if "wealth" in attribute or "$" in t:  # dollar ranges -> upper magnitude
        nums = [
            float(val.replace(",", ""))
            * {"million": 1e6, "thousand": 1e3, "k": 1e3}.get(u, 1.0)
            for val, u in re.findall(
                r"\$?([\d,]+(?:\.\d+)?)\s*(million|thousand|k)?", t
            )
        ]
        return max(nums) if nums else None
    # any *_degree attr
    return next((float(v) for kw, v in _DEGREE_SCALE if kw in t), None)


def expected_value(probs, values):
    """Prob-weighted mean attribute value over substantive (non-None) choices."""
    keep = [(p, v) for p, v in zip(probs, values) if v is not None]
    w = np.array([p for p, _ in keep])
    v = np.array([v for _, v in keep])
    return float((w * v).sum() / w.sum()) if w.sum() > 0 else float("nan")


# %% Setup: NDIF, store, personas
load_dotenv()

torch.set_grad_enabled(False)
OUT.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = OUT / f"results_{MODEL_NAME.split('/')[-1]}.json"

store = HFPersonaVectorStore(
    REPO_ID, MODEL_NAME, mask_strategy=MaskStrategy.ANSWER_MEAN
)

all_ids = store.list_personas(["templated"], include_baseline=False)
n_layers = int(store.load("templated", all_ids[0]).shape[0])
candidate_layers = list(range(8, n_layers, max(3, n_layers // 12)))
dataset = SynthPersonaDataset()
eval_ids = [pid for pid in all_ids if pid in set(dataset.persona_ids)][:N_PERSONAS]
print(
    f"{MODEL_NAME}: {len(all_ids)} personas, {n_layers} layers; eval {len(eval_ids)}, attrs={ATTRS}"
)

# %% Build one act-add direction per attribute (offline, no NDIF)
directions = {}
for attr in ATTRS:
    info = build_attribute_direction(
        store,
        dataset,
        attr,
        variant="templated",
        candidate_layers=candidate_layers,
        persona_ids=all_ids,
    )
    directions[attr] = info
    print(
        f"{attr:24s} layer={info['layer']:2d} AUC={info['auc']:.3f} "
        f"gap_norm={info['gap_norm']:.2f} act_norm={info['act_norm']:.1f}"
    )

# %% Load live model + scoring helpers
model = StandardizedTransformer(MODEL_NAME)
LETTERS = ["A", "B", "C", "D", "E", "F"]


def substantive(attribute, qa):
    """Drop the unsure option; return (letters, options, values, letter_token_ids)."""
    opts = [
        str(c).strip()
        for c in qa.choices
        if choice_value(attribute, str(c)) is not None
    ]
    values = [choice_value(attribute, o) for o in opts]
    letters = LETTERS[: len(opts)]
    tok_ids = [
        model.tokenizer(letter, add_special_tokens=False).input_ids[0]
        for letter in letters
    ]
    return letters, opts, values, tok_ids


def render_mcq(qa, letters, opts):
    body = "\n".join(f"{letters[i]}. {opts[i]}" for i in range(len(opts)))
    user = f"{MC_INSTRUCTION}\n\nQUESTION:\n{qa.question.strip()}\n\nOPTIONS:\n{body}\n\nANSWER:"
    return model.tokenizer.apply_chat_template(
        [{"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )


def score(prompt, letter_ids, layer, vec, factor):
    """P(letter) over the substantive choices at the answer position, steered by `factor`."""
    ids = model.tokenizer(
        prompt, return_tensors="pt", add_special_tokens=False
    ).input_ids
    pos = ids.shape[1] - 1
    for attempt in range(4):  # NDIF drops are intermittent; retry
        try:
            with model.trace(ids, remote=True):
                if factor:  # steer ALL positions, like free-form generation
                    model.steer(layers=layer, steering_vector=vec, factor=float(factor))
                lp = torch.log_softmax(
                    model.logits[0, pos, letter_ids].float(), dim=-1
                ).save()
            return lp.exp().detach().cpu().numpy()
        except Exception:
            if attempt == 3:
                raise
            time.sleep(8)


# %% Sweep: each persona x attribute, score the choice distribution per t
results = []
for pid in eval_ids:
    qa_by_attr = {
        q.qid.split("explicit_seed_attribute_")[-1]: q
        for q in dataset.get_qa(pid)
        if q.item_type == "mcq" and "explicit_seed_attribute_" in q.qid
    }
    for attr in ATTRS:
        qa = qa_by_attr.get(attr)
        if qa is None:
            continue
        d = directions[attr]
        letters, opts, values, letter_ids = substantive(attr, qa)
        prompt = render_mcq(qa, letters, opts)
        try:
            for t in TS:
                probs = score(
                    prompt,
                    letter_ids,
                    d["layer"],
                    d["unit_direction"],
                    steering_coefficient(d, t),
                )
                results.append(
                    {
                        "persona": pid,
                        "attribute": attr,
                        "t": t,
                        "layer": d["layer"],
                        "options": opts,
                        "values": values,
                        "probs": probs.round(4).tolist(),
                        "exp_value": expected_value(probs, values),
                    }
                )
        except (
            Exception
        ) as e:  # persistent NDIF failure for this item: skip, keep going
            print(f"  SKIP {pid[:8]} {attr}: {type(e).__name__}: {e}")
            continue
        print(f"done {pid[:8]} {attr}")
    RESULTS_PATH.write_text(json.dumps(results, indent=2))  # incremental save
print(f"saved {RESULTS_PATH} ({len(results)} rows)")


# %% ## Headline: does steering move the answer?
# `norm.slope` = (exp_value@+t - exp_value@-t) / value-span, comparable across
# attributes. Positive => +steering pushes the choice toward higher values.
def agg(attr, t):
    xs = [
        r["exp_value"]
        for r in results
        if r["attribute"] == attr and r["t"] == t and not np.isnan(r["exp_value"])
    ]
    return float(np.mean(xs)) if xs else float("nan")


print(
    f"\n{'attribute':24s} {'exp@-a':>10s} {'exp@0':>10s} {'exp@+a':>10s} {'norm.slope':>11s}"
)
for attr in ATTRS:
    vals = [v for r in results if r["attribute"] == attr for v in r["values"]]
    if not vals:
        continue
    span = (max(vals) - min(vals)) or 1.0
    lo, mid, hi = agg(attr, TS[0]), agg(attr, 0.0), agg(attr, TS[-1])
    print(f"{attr:24s} {lo:10.2f} {mid:10.2f} {hi:10.2f} {(hi - lo) / span:+11.2f}")
