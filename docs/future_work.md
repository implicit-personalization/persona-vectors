# Future Work

Ideas that are deliberately **not** implemented yet, parked here so the core
stays simple.

## Modulating the steering coefficient over generation steps

Today steering uses a single coefficient applied at every generated position
(`tracer.all()` in `generate_steered_once`). A natural extension is to vary the
coefficient *across generation steps* — strong while a property is being
established, then tapering so the continuation stays fluent.

**Do not hand-pick a fixed schedule.** The right reference is Scalena, Sarti &
Nissim, *Multi-property Steering of Large Language Models with Dynamic Activation
Composition* (BlackboxNLP 2024, <https://aclanthology.org/2024.blackboxnlp-1.34/>).
Their **Dynamic Activation Composition** chooses the per-step intensity
*adaptively* with an information-theoretic (KL-divergence) criterion rather than a
fixed ramp, and composes multiple property vectors at once. Any per-step steering
here should follow that approach instead of an invented schedule.

### Parked prototype (a fixed schedule — only a starting point)

This was prototyped and removed to avoid over-complicating `steer_generate.py`.
It uses `tracer.iter[step]` to apply a per-step factor. Keep it only as a wiring
reference; replace the fixed `schedule` with the adaptive criterion above before
using it for anything real.

```python
# In generate_steered_once, replacing the single `tracer.all()` block:
#   schedule: Callable[[int], float] | None = None   # added to the signature
#   (requires max_new_tokens to bound the step loop)
with model.generate(prompt, remote=remote, backend=backend, **generation_kwargs) as tracer:
    if schedule is not None and layer is not None:
        for step in range(int(generation_kwargs["max_new_tokens"])):
            step_factor = schedule(step)
            if step_factor:
                with tracer.iter[step]:   # nnsight: scope the hook to one step
                    model.steer(layers=layer, steering_vector=steering_vector,
                                factor=float(step_factor))
    elif factor and layer is not None:
        with tracer.all():
            model.steer(layers=layer, steering_vector=steering_vector, factor=float(factor))
    out = model.generator.output.save()

# generate_steered would pass `schedule=lambda step, f=factor: f * schedule(step)`
# so the swept factor scales the shape and factor=0 stays the baseline.
```

## Full per-position steering vectors

Per-*position* vectors (steering a different direction at each token) would
require changing the activation artifact from `(num_layers, hidden)` to
`(num_layers, seq_len, hidden)` and can't be averaged across variable-length
personas. Approximate with the existing segment masks
(`ANSWER_FIRST/LAST/MEAN`, `QUESTION_LAST`, `PERSONA_LAST`) first; only build full
per-position storage if those prove insufficient. Multi-vector composition is
again covered by the paper above.

## Trait vectors on the Hub

Trait vectors load locally only today. When published, add an
`HFTraitVectorStore` mirroring `HFPersonaVectorStore` (and a `TraitVectorSource`
union over the two) so [trait vectors](traits.md) load from the Hub or disk
through one contract.
