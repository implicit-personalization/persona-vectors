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
