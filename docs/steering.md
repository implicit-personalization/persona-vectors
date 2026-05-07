# Steering

Compute persona steering vectors from pre-extracted activations for steering
experiments toward a specific persona.
Core module: `src/persona_vectors/steering.py`

---

## Pipeline

```
Extract Activations (`notebook_extract.py`) → Compute Steering Vector → Save
```

Steering reuses the activations extracted by `notebook_extract.py`. No
re-extraction is needed.

---

## Quick Start

```python
from persona_vectors.steering import compute_steering_vector, save_steering_vector

sv_dict = compute_steering_vector(
    persona_id="0023952f-142e-434b-82e2-7a7451b7c55f",
    model_name="google/gemma-2-9b-it",
    layer_idx=20,
    mask_strategy="answer_previous",
)

save_steering_vector(sv_dict, "artifacts/vectors/my_persona")
```

### CLI

```bash
uv run python main.py steer --persona-id <UUID> --model google/gemma-2-9b-it --layer 20
uv run python main.py steer --persona-id <UUID> --model google/gemma-2-9b-it --layer 20 --mask-strategy answer_previous
```

---

## Method: Contrastive Mean-Diff

For each persona, two prompt variants are extracted:

- **Positive (biography):** Full persona biography as system prompt + QA
- **Negative (templated):** Generic templated prompt + QA

Extraction already averages the masked hidden states across QA pairs and
masked tokens, so each saved artifact is a single `(num_layers, hidden_size)`
tensor. The default mask is `answer_mean` (every assistant-answer token);
`answer_previous` uses the position immediately before the first answer
token. At a chosen layer the steering vector is:

```
steering_vector = biography_h[layer] - templated_h[layer]
```

Adding this vector to the residual stream at inference shifts model behavior
toward the persona.

---

## Functions

### compute_steering_vector()

```python
sv_dict = compute_steering_vector(
    persona_id="...",
    model_name="google/gemma-2-9b-it",
    layer_idx=20,
    mask_strategy="answer_previous",
    activations_dir="artifacts/activations",
)
```

Returns a dict with:
- `steering_vector`: shape `[1, 1, d_model]`
- `suggested_alpha`: scaling coefficient (`20 * mean_rms / ||sv||`), where
  `mean_rms` is the RMS of the negative (templated) mean activation at the
  chosen layer
- `persona_id`, `layer`, `model_id`, `hidden_size`: metadata

### save_steering_vector()

Saves a directory containing a safetensors file plus metadata:

```python
save_steering_vector(sv_dict, "artifacts/vectors/my_persona")
# Creates:
#   artifacts/vectors/my_persona/steering_vector.safetensors
#   artifacts/vectors/my_persona/metadata.json
```

### load_steering_vector()

```python
from persona_vectors.steering import load_steering_vector

sv_dict = load_steering_vector("artifacts/vectors/my_persona")
sv = sv_dict["steering_vector"]  # [1, 1, d_model]
alpha = sv_dict["suggested_alpha"]
```

---

## Output Format

```
artifacts/vectors/{persona_id}/
├── steering_vector.safetensors   # steering_vector tensor
└── metadata.json                 # suggested_alpha, layer, model_id, hidden_size
```

The `metadata.json` file also records `suggested_alpha`.

When using a non-default extraction strategy, pass the same `mask_strategy` to
`compute_steering_vector()` or `main.py steer` so it loads the matching
activation artifacts.

---

## Choosing a Layer

Mid layers (around 15-25 for Gemma-2-9b-it with 42 layers) typically work best
for persona steering. Use `notebook_steer.py` to experiment with different layers
and inspect the vector norm and suggested alpha.
