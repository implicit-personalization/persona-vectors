# Trait Vectors

A persona vector mixes every attribute a persona carries, so a population
difference-of-means direction for one attribute absorbs whatever co-occurs with
it. A **trait vector** isolates a single attribute with a minimal pair: swap only
that attribute on a persona, extract both views, and average the *within-pair*
activation delta. Everything that did not change cancels.

Core module: `src/persona_vectors/traits.py`

## How it works

1. `persona_data.templated.swap_attribute` edits one attribute and re-renders the
   whole templated view (so coupled sentences like age+sex stay coherent). Binary
   attributes auto-flip.
2. For each persona, extract activations for the original and swapped views
   (reusing the extraction pipeline), then take `act(value_to) − act(value_from)`.
   Activations are oriented **by attribute value**, not by which value was
   original, so a Male→Female swap and a Female→Male swap reinforce instead of
   cancelling on average.
3. Average the paired deltas over personas → the per-layer trait vector.

This builds the **description-level** flavor (`PERSONA_MEAN` over the swapped
templated view) for **binary** attributes; the answer-level flavor
(`ANSWER_MEAN`, force-decoded explicit answer) and non-binary attributes reuse
the same orientation logic.

## Extract and build

```python
from persona_vectors.extraction import MaskStrategy
from persona_vectors.traits import extract_trait_deltas, build_trait_direction

runs = [(p, dataset.train_test_split(p.id, n_train=1)[0]) for p in dataset]
runs = [(p, qa) for p, qa in runs if qa]  # the QA only builds the prompt

deltas = extract_trait_deltas(
    model, dataset, "sex", runs,
    variant="templated", mask_strategy=MaskStrategy.PERSONA_MEAN, remote=False,
)
info = build_trait_direction(deltas, candidate_layers=[13])
```

`build_trait_direction` returns the **same dict schema** as
[`build_attribute_direction`](steering.md) (`layer`, `unit_direction`,
`gap_norm`, `auc`, `positive`, …), so a trait vector drops straight into the
steering harness via `build_steering_spec` / `generate_steered`.

## Steering

A trait `info` dict feeds `build_steering_spec` / `generate_steered` unchanged
(see [Steering](steering.md)).

## Notebook

`notebooks/notebook_extract_trait.py` runs the full flow: extract a trait vector
per binary attribute, then compare the **trait-cosine** matrix against the
**co-occurrence** (Cramér's V) matrix — high co-occurrence with low trait-cosine
means the minimal-pair extraction successfully deconfounded that pair.
