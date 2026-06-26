# Steering

Steering is experimental. It computes a persona direction from saved activations:

```text
steering_vector = biography[layer] - templated[layer]
```

Core module: `src/persona_vectors/steering.py`

## CLI

```bash
uv run python main.py steer \
  --model google/gemma-2-9b-it \
  --persona-id <UUID> \
  --layer 20 \
  --mask-strategy answer_mean
```

Use the same `--mask-strategy` that was used during extraction.

## API

```python
from persona_vectors.steering import (
    compute_steering_vector,
    load_steering_vector,
    save_steering_vector,
)

sv = compute_steering_vector(
    persona_id="<UUID>",
    model_name="google/gemma-2-9b-it",
    layer_idx=20,
    mask_strategy="answer_mean",
)

save_steering_vector(sv, "artifacts/vectors/<UUID>")
loaded = load_steering_vector("artifacts/vectors/<UUID>")
```

`compute_steering_vector()` returns:

- `steering_vector`: tensor with shape `(1, 1, hidden_size)`
- `suggested_alpha`: `20 * mean_rms / ||sv||`
- `persona_id`, `layer`, `model_id`, `hidden_size`

## Output

```text
artifacts/vectors/<persona_id>/
├── steering_vector.safetensors
└── metadata.json
```

Mid layers are usually the first place to try, but layer choice is model and task dependent. Use `notebooks/notebook_steer.py` for experiments.

Steering is applied uniformly at every generated position (`tracer.all()`).
Modulating the coefficient *over generation steps* is deferred — see
[Future work](future_work.md).

## Steering during generation

To add a direction on the *live* model and read off the behavioral shift, use
`generate_steered` with a direction dict (e.g. from
[`build_trait_direction`](traits.md)):

```python
from persona_vectors.steering import generate_steered, steering_coefficient

out = generate_steered(
    model,
    "Tell me about where you grew up.",
    info["layer"],
    info["unit_direction"],
    [0.0, steering_coefficient(info, 4.0), steering_coefficient(info, -4.0)],
    system="You are a human being having a casual conversation.",
    max_new_tokens=120,
    remote=False,
)  # -> {factor: continuation}
```

`steering_coefficient(info, strength)` calibrates the push in **gap units**
(`strength=1` lands the activation at the opposite-class centroid); `strength=0`
is the unsteered baseline.

## Band + adaptive steering (recommended)

A single-layer push is weak in-distribution. The strongest, still-on-manifold
method steers a **band** of layers — each with its own trait direction at a modest
per-layer strength — optionally modulating intensity over generation steps.

```python
from persona_vectors.artifacts import TraitVectorStore
from persona_vectors.traits import load_trait_band
from persona_vectors.steering import (
    band_steering_vectors, generate_band_steered, dim_schedule,
)

store = TraitVectorStore("google/gemma-2-9b-it")
band = load_trait_band(store, "age", layers=range(14, 31))   # {layer: direction}
vectors = band_steering_vectors(band, strength=1.0)          # strength=1 = opposite centroid

# constant (drop-in for fixed steering, just multi-layer and stronger)
text = generate_band_steered(model, prompt, vectors, system=SYS, max_new_tokens=120)

# adaptive: pass the schedule bare; it's called as schedule(step, max_new_tokens)
text = generate_band_steered(model, prompt, vectors, schedule=dim_schedule, ...)
```

Pass a **list** of bands to `band_steering_vectors` to compose several traits
(summed per layer; correlated traits reinforce). `dim_schedule` tapers intensity
1→0 to keep long, hard-steered generations fluent; `start_schedule` steers only
the opening tokens. You keep the strength dial — the schedule just shapes it.
