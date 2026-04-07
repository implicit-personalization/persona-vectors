# Artifacts

Save, load, and query activation vectors via `ActivationStore`. This is the second
step in the persona vectors pipeline — after extracting activations with
`extract_activations()`, you save them here, then later load them for comparison and
analysis.

```
Dataset → Format Prompts → Extract Activations → Save → Analyze/Compare
```

---

## Why save activations?

Extracting activations is expensive — it requires running the model forward pass.
By saving activations to disk:

- **Reuse:** Run extraction once, analyze many times with different methods
- **Share:** Share activations with collaborators without them needing the model
- **Iterate:** Try different analysis techniques without re-extraction

---

## Directory Structure

```bash
artifacts/activations/
└── google__gemma-2-2b-it/                # model name with / replaced by __
    └── templated/                        # prompt_variant
        └── persona_001/                  # persona_id
            ├── activations.safetensors   # tensor (n_questions, n_layers, d_model)
            └── metadata.json             # {"persona_id", "persona_name", "questions"}
```

The root directory defaults to `$ARTIFACTS_DIR/activations` (falls back to
`./artifacts/activations` when the env var is not set).

---

## ActivationStore

Save/load goes through `ActivationStore`, which binds a `model_name` and
`root_dir` together so they don't need to be repeated at every call site.
Query helpers like `list_personas`, `list_layers`, and `load_mean_activations`
are top-level functions in `persona_vectors.artifacts`.

```python
from persona_vectors.artifacts import ActivationStore

# Uses $ARTIFACTS_DIR/activations (or ./artifacts/activations) by default
store = ActivationStore("google/gemma-2-2b-it")

# Or pass an explicit root
store = ActivationStore("google/gemma-2-2b-it", root_dir="./my_artifacts/activations")
```

### Save

```python
store.save(
    prompt_variant="templated",
    persona_id=persona.id,
    persona_name=persona.name,
    per_question_vectors=activations,  # (n_questions, n_layers, d_model)
    questions=questions,               # list of question strings
)
```

### Load

```python
vectors, questions = store.load("templated", persona.id)
# vectors: (n_questions, n_layers, d_model) tensor
# questions: list of question strings
```

### Query available data

```python
from persona_vectors.artifacts import list_layers, list_personas, load_persona_names, load_mean_activations

# Persona ids present for every requested variant
persona_ids = list_personas(store.root_dir, store.model_name, ["templated", "biography"])

# Layer indices shared across all variant/persona combinations
layers = list_layers(store.root_dir, store.model_name, ["templated"], persona_ids)

# Display names from saved metadata
names = load_persona_names(store.root_dir, store.model_name, ["templated"], persona_ids)
# {"persona_001": "Alice", ...}
```

### Load mean activations for cosine comparison

```python
traces, names, errors = load_mean_activations(
    store.root_dir, store.model_name, persona_ids, variant_a="templated", variant_b="biography"
)
# traces: list of (persona_id, mean_vectors_a, mean_vectors_b)
```

---

## File Format

- **activations.safetensors**: Efficient binary format using PyTorch's safetensors,
  supports memory-mapped loading for large datasets.
- **metadata.json**: Human-readable JSON with persona metadata and question strings.

---

## Utility functions

```python
from persona_vectors.artifacts import model_dir_name

model_dir_name("google/gemma-2-2b-it")  # "google__gemma-2-2b-it"
```

`model_dir_name` maps model identifiers to filesystem-safe directory names (preserves
case and hyphens, replaces `/` with `__`). UI/export filename slugging now lives in
`persona-ui/utils/helpers.py`.
