# Artifacts

`PersonaVectorStore` reads and writes local tensors. `HFPersonaVectorStore` reads published Hub datasets. The on-disk layout below matches what `HFPersonaVectorStore` expects on the Hub, so the same model code reads either source.

Core module: `src/persona_vectors/artifacts.py`

`PersonaVectorStore` and `TraitVectorStore` share a `LocalVectorStore` base that
owns the local safetensors + manifest machinery
(`<root>/<model>/<mask_strategy>/<variant>/<item>.safetensors`); they differ only
in the item key (persona vs attribute) and per-item metadata.

## Layout

```text
artifacts/activations/   # also: artifacts/persona-vectors/ from the all-questions script
└── google__gemma-2-9b-it/
    └── answer_mean/
        └── biography/
            ├── manifest.json
            └── <persona_id>.safetensors
```

Each safetensors file contains one `activations` tensor with shape `(num_layers, hidden_size)` — the persona vector for that variant, averaged across QA pairs and selected tokens. The manifest stores shape metadata plus persona names and QA sample ids.

`artifacts/activations/` is the default. `scripts/extraction_all_questions.sh` writes under `artifacts/persona-vectors/` to keep all-questions runs separate from train-split runs; pass `--activations-dir artifacts/persona-vectors` (or `root_dir=...` to the store) to read it back.

Probe training writes a separate tree under `artifacts/probes/`. Each saved
probe directory contains the canonical `probe.json` + `weights.safetensors`
bundle. Use `persona_vectors.probes.load_probe_artifact(...)` to read that
bundle without re-implementing the schema. See [Probes](probes.md) for the
probe artifact contract.

## Local Store

```python
from persona_vectors.artifacts import PersonaVectorStore

store = PersonaVectorStore("google/gemma-2-9b-it", mask_strategy="answer_mean")

vectors = store.load("biography", "<persona_id>")
persona_ids = store.list_personas(["biography"])
names = store.persona_names(persona_ids, variants=["biography"])
available = store.available_variants(["biography", "templated"])
layers = store.list_layers(["biography"], persona_ids)
```

`list_personas(["biography", "templated"])` returns only personas present in
both variants. This keeps variant comparisons aligned. `list_personas()`
excludes `baseline_assistant` by default; pass `include_baseline=True` when the
baseline should remain visible.

To save:

```python
store.save(
    prompt_variant="biography",
    persona_id=persona.id,
    persona_name=persona.name,
    vectors=activations,
    sample_ids=[qa.qid for qa in qa_pairs],
)
```

## Hub Store

```python
from persona_vectors.artifacts import HFPersonaVectorStore

store = HFPersonaVectorStore(
    "implicit-personalization/synth-persona-vectors",
    "google/gemma-2-9b-it",
    mask_strategy="answer_mean",
)

variant = store.available_variants(["biography", "templated"])[0]
vectors = store.load(variant, "<persona_id>")
layers = store.list_layers([variant], ["<persona_id>"])
```

Hub datasets use one config per `<model_dir>__<mask_strategy>` and one split per
prompt variant. `HFPersonaVectorStore` is read-only and supports the same discovery
methods as the local store: `load`, `available_variants`, `list_personas`,
`persona_names`, and `list_layers`.

`HFPersonaVectorStore.release_cache()` clears cached datasets and metadata.

## Trait Vector Store

`TraitVectorStore` stores minimal-pair [trait vectors](traits.md): one tensor per
**attribute** — the per-layer mean swap delta `(num_layers, hidden_size)` — so a
steering direction can be rebuilt at any layer. It lives under `trait_vectors/`
so it never collides with persona activations.

```text
artifacts/trait_vectors/
└── google__gemma-2-2b-it/
    └── persona_mean/
        └── templated/
            ├── manifest.json    # per-attribute: positive, value_from/to,
            │                    # n_personas, auc_by_layer, act_norm_by_layer
            └── sex.safetensors   # (num_layers, hidden) mean delta
```

```python
from persona_vectors.artifacts import TraitVectorStore
from persona_vectors.traits import save_trait_deltas, load_trait_direction

store = TraitVectorStore("google/gemma-2-2b-it", mask_strategy="persona_mean")
save_trait_deltas(store, deltas, mask_strategy="persona_mean")

info = load_trait_direction(store, "sex", layer=13)   # steering-ready dict
attrs = store.list_attributes(variant="templated")
```

Trait vectors load locally today; an `HFTraitVectorStore` (mirroring
`HFPersonaVectorStore`) can be added when they are published to the Hub.

## Analysis-facing bundle

The stores stay intentionally low-level. When analysis code needs vectors plus
their aligned ids, names, and shared layers, build one dataset-facing bundle:

```python
from persona_vectors.analysis import load_analysis_dataset

dataset = load_analysis_dataset(store, ["biography", "templated"])
dataset.persona_ids
dataset.persona_names
dataset.layers
dataset.samples("biography")
```

This keeps local and Hub reads on the same contract while avoiding repeated
metadata calls in downstream tools.

## Publishing

```bash
uv run python main.py push \
  --model google/gemma-2-9b-it \
  --repo implicit-personalization/synth-persona-vectors
```

Python callers can use `persona_vectors.hub.push_to_hub(...)` directly.

## Helpers

`model_dir_name("google/gemma-2-9b-it")` returns `"google__gemma-2-9b-it"`.

Use `discover_activation_models(root_dir, mask_strategy)` to list local model ids
that have at least one saved artifact for a mask strategy.

Use Hub discovery helpers when building model pickers or notebooks:

```python
from persona_vectors.hub import list_hub_vector_models, parse_vector_config_name

models_by_mask = list_hub_vector_models(
    "implicit-personalization/synth-persona-vectors"
)
parsed = parse_vector_config_name("google__gemma-2-9b-it__answer_mean")
```
