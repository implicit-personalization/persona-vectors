# Future Work

What is implemented, what was tested and dropped, and what remains.

## Per-step intensity scheduling — implemented

`generate_band_steered` takes an optional `schedule(step, n_steps) -> [0, 1]` multiplier on
the user's base strength, applied per generation step via `tracer.iter`.
`dim_schedule` (linear taper) and `start_schedule` (steer the opening tokens) keep
hard-steered generations fluent while preserving the strength dial. These are the
remote-feasible schedules from Scalena, Sarti & Nissim, *Multi-property Steering
with Dynamic Activation Composition* (BlackboxNLP 2024,
<https://aclanthology.org/2024.blackboxnlp-1.34/>).

**Full KL-adaptive DAC (their Eq. 6) — tested, not adopted.** It sets intensity
fully automatically from a KL criterion (no dial) and is a per-token feedback loop,
so it is local-only and cannot run on the remote path. The simple, dial-preserving
schedules above deliver the same fluency benefit, so DAC is not kept in the core.

## Per-position direction extraction — tested, not adopted

The paper extracts a separate direction per generation step. We tested this
(K=50 minimal pairs, per-position contrast at the answer tokens): the per-position
directions drift from each other but largely as noise, each still tracks the pooled
direction, and there is no evidence they steer better than the single pooled vector
across a layer band. For a *global* persona attribute, per-position extraction adds
a heavier pipeline for no gain. See the report appendix.

## Trait vectors on the Hub

Local persistence is implemented (`TraitVectorStore` + `traits.save_trait_deltas` /
`load_trait_direction` / `load_trait_band`, reading the per-layer mean delta from
`artifacts/trait_vectors/`). Only Hub loading remains: add an `HFTraitVectorStore`
mirroring `HFPersonaVectorStore` (and a `TraitVectorSource` union over the two) so
[trait vectors](traits.md) load from the Hub or disk through one contract.
