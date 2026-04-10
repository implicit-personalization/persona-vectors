# Artifacts

Store and load activation tensors with `ActivationStore`.
The default root is `$ARTIFACTS_DIR/activations`, or `./artifacts/activations` if the env var is not set.
Core module: `src/persona_vectors/artifacts.py`

## Directory Layout

```bash
artifacts/activations/
└── google__gemma-2-2b-it/
    └── templated/
        └── persona_001/
            ├── activations.safetensors
            └── metadata.json
```

## ActivationStore

```python
from persona_vectors.artifacts import ActivationStore

store = ActivationStore("google/gemma-2-2b-it")

store.save(
    prompt_variant="templated",
    persona_id=persona.id,
    persona_name=persona.name,
    per_question_vectors=activations,
    questions=questions,
)

vectors, questions = store.load("templated", persona.id)
```

## Query Helpers

- `list_personas()`: persona ids available across variants
- `list_layers()`: shared layer indices
- `load_persona_names()`: display names from saved metadata
- `load_mean_activations()`: mean vectors for comparing two variants

`store.save()` writes both `activations.safetensors` and `metadata.json`.

## File Format

- `activations.safetensors`: tensor with shape `(n_questions, n_layers, d_model)`
- `metadata.json`: persona metadata, saved questions, and tensor shape fields

## Utility

`model_dir_name("google/gemma-2-2b-it")` returns `"google__gemma-2-2b-it"`.
