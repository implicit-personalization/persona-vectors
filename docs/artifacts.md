# Artifacts

Store and load activation tensors with `ActivationStore`.
The default root is `$ARTIFACTS_DIR/activations`, or `./artifacts/activations` if the env var is not set.
Core module: `src/persona_vectors/artifacts.py`

## Directory Layout

```bash
artifacts/activations/
└── google__gemma-2-2b-it/
    └── answer_mean/
        └── templated/
            ├── manifest.json
            └── <persona_id>.safetensors
```

Artifacts are grouped by model, mask strategy, and prompt variant. A single
`manifest.json` tracks all personas available for that grouping, while each
persona has its own safetensors file next to the manifest.

The Assistant baseline is a persona-less artifact variant. It is stored once
under `baseline/` with the shared baseline persona id from
`persona_data.prompts` and can be added as an Assistant reference in
persona-space comparisons.

## ActivationStore

```python
from persona_vectors.artifacts import ActivationStore

store = ActivationStore("google/gemma-2-2b-it")

store.save(
    prompt_variant="templated",
    persona_id=persona.id,
    persona_name=persona.name,
    per_question_vectors=activations,
    sample_ids=[qa.qid for qa in qa_pairs],
    mask_strategy="answer_previous",
)

vectors, sample_ids = store.load(
    "templated",
    persona.id,
    mask_strategy="answer_previous",
)
```

## Query Helpers

- `list_personas()`: persona ids available across all requested variants for a mask strategy
- `list_layers()`: shared layer indices
- `load_persona_names()`: display names from saved metadata
- `load_mean_activations()`: mean vectors for comparing two variants

All query helpers accept `mask_strategy`, defaulting to `answer_mean`.
`store.save()` writes one persona safetensors file and updates `manifest.json`.

When `list_personas()` is called with multiple variants, it returns only personas
present in every requested variant. If some personas exist for only a subset of
those variants, it warns that they were skipped.

## File Format

- `<persona_id>.safetensors`: one `activations` tensor with shape `(n_samples, n_layers, d_model)`
- `manifest.json`: tensor shape fields and per-persona `name`/`sample_ids`

## Utility

`model_dir_name("google/gemma-2-2b-it")` returns `"google__gemma-2-2b-it"`.
