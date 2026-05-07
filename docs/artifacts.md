# Artifacts

Store and load activation tensors with `ActivationStore`, or read a published
Hugging Face dataset with `HFActivationStore`.
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

The Assistant baseline is stored like any other persona, under the same prompt
variant directories, with persona id `baseline_assistant`.

## ActivationStore

```python
from persona_vectors.artifacts import ActivationStore

store = ActivationStore(
    "google/gemma-2-2b-it",
    mask_strategy="answer_previous",
)

store.save(
    prompt_variant="templated",
    persona_id=persona.id,
    persona_name=persona.name,
    vectors=activations,
    sample_ids=[qa.qid for qa in qa_pairs],
)

vectors = store.load(
    "templated",
    persona.id,
)

available_variants = store.available_variants(
    ["templated", "biography"],
)
persona_ids = store.list_personas(available_variants)
persona_names = store.persona_names(
    persona_ids,
    variants=available_variants,
)
```

## Query Helpers

Use the `ActivationStore` methods for common discovery:

- `store.available_variants()`: candidate variants with at least one saved persona
- `store.list_personas()`: persona ids available across all requested variants
- `store.persona_names()`: display names from saved metadata

`ActivationStore` defaults to `answer_mean`; pass `mask_strategy=...` once when
constructing the store, or override it on an individual method call.
By default, `available_variants()` checks `PERSONA_VARIANTS`; pass
`variants=...` to the store constructor if a workflow should use a different
candidate set.
`store.persona_names(persona_ids)` returns names in the input persona order, so
`list(names.values())` is stable on supported Python versions.

When listing personas across multiple variants, the result includes only
personas present in every requested variant. If some personas exist for only a
subset of those variants, it warns that they were skipped.

## Hugging Face Dataset Store

`HFActivationStore` reads an already-published activation dataset built by
`scripts/push_to_hf.py`. It does not write local files and does not push data.

```python
from persona_vectors.artifacts import HFActivationStore

store = HFActivationStore(
    "implicit-personalization/synth-persona-vectors",
    "google/gemma-2-9b-it",
    mask_strategy="answer_mean",
)

available_variants = store.available_variants(["biography", "templated"])
variant = available_variants[0]
vectors = store.load(variant, "<persona_id>")
persona_ids = store.list_personas([variant])
names = store.persona_names(persona_ids)
```

The Hub layout mirrors the local grouping: one dataset config per
`<model_dir>__<mask_strategy>` and one split per prompt variant. The Hub store
is read-only, but exposes the same discovery methods as
`ActivationStore`: `load`, `available_variants`, `list_personas`, and
`persona_names`.
Ask for variants in preference order if the published dataset does not have
every local prompt variant yet.

For a complete PCA/similarity example using the published dataset directly, see
`notebooks/notebook_hf_compare.py`.

## File Format

- `<persona_id>.safetensors`: one `activations` tensor with shape `(num_layers, hidden_size)`
- `manifest.json`: tensor shape fields and per-persona `name`/`sample_ids`

Extraction averages across QA pairs before saving. `sample_ids` remain in the
manifest for provenance, but `ActivationStore.load()` returns only the saved
activation tensor.

## Utility

`model_dir_name("google/gemma-2-2b-it")` returns `"google__gemma-2-2b-it"`.
