# Persona Vectors

[![Docs](https://img.shields.io/badge/docs-view-purple?logo=github)](https://implicit-personalization.github.io/persona-vectors/)
[![PyPI](https://img.shields.io/pypi/v/persona-vectors?logo=pypi&label=PyPI)](https://pypi.org/project/persona-vectors/)

Extract persona-aligned activation vectors from language models and analyze how persona prompts move hidden states.

> [!WARNING]
> This is very experimental currently 🚨

## Overview

Given a set of personas and evaluation questions, this project:

1. Formats each persona as a system prompt (short `templated` or long `biography`)
2. Extracts hidden states at each layer with configurable token masking
3. Averages masked hidden states across QA pairs and saves one persona-level vector per layer

The resulting vectors can be compared across layers (cosine similarity) and eventually used for steering experiments.

## Repository Layout

```
persona-vectors/
├── notebooks/
│   ├── notebook_extract.py      # Extraction pipeline (primary working script)
│   ├── notebook_pca.py          # PCA scree + 3D PCA colored by clustering labels
│   ├── notebook_similarity.py   # Persona similarity heatmap, dendrograms, variant cosine
│   └── notebook_steer.py        # Steering experiments
├── src/persona_vectors/
│   ├── activations.py           # Core extraction helpers
│   ├── analysis.py              # PCA / UMAP projections and scatter plots
│   ├── artifacts.py             # Local and Hugging Face activation artifact stores
│   ├── preview.py               # Token-mask preview helpers for CLI/UI rendering
│   ├── plots.py                 # Plotly figures for layer-wise analysis
│   ├── steering.py              # Steering vector computation and application
│   └── parser.py                # CLI argument parsing
├── artifacts/                   # Saved activations (gitignored)
├── docs/                        # Reference documentation
└── main.py                      # CLI entry point
```

Dataset loading (`SynthPersonaDataset`) and environment helpers come from the
sibling [persona-data](../persona-data) package.

For local development, uncomment the `path` source in `pyproject.toml` and keep
`persona-data` checked out next to this repo.

## Installation

```bash
uv sync
cp .env.example .env
```

Python `>=3.12` is required.

## Quickstart

```bash
# Extract activations (run this first)
uv run python -m notebooks.notebook_extract

# PCA + clustering colorings (Hub artifacts; uncomment in-file for local store)
uv run python -m notebooks.notebook_pca

# Pairwise similarity, dendrograms, and prompt-variant cosine views
uv run python -m notebooks.notebook_similarity

# Build interactive persona-vector PCA and similarity plots from saved activations
uv run python main.py analyze --model google/gemma-2-9b-it --variant biography --mask-strategy answer_mean

# Compute a steering vector from saved activations
uv run python main.py steer --persona-id <UUID> --model google/gemma-2-9b-it --layer 20
```

## Streamlit App

The Streamlit UI lives in the sibling [persona-ui](../persona-ui) repo.

## How It Works

### Notebooks

`notebook_extract.py` runs a small end-to-end extraction example:

1. Load dataset questions and answers
2. Build masks for the selected token spans
3. Extract activations and average them across QA pairs
4. Save the persona-level activation tensor to disk

`notebook_pca.py` and `notebook_similarity.py` both use `HFActivationStore` by
default to load the published Hub dataset, with commented lines for switching
to local `ActivationStore` artifacts. The first focuses on 3D PCA scatter
views colored by k-means / agglomerative / HDBSCAN clusters; the second covers
pairwise persona similarity, hierarchical dendrograms with selectable linkage,
and prompt-variant cosine comparisons. Clustering and dendrograms use centered,
unit-normalized inputs by default so labels reflect vector direction more than
raw magnitude.

`notebook_steer.py` loads saved activations and computes a steering vector for a
selected persona.

### Saved Format

Each extraction produces:

```
artifacts/activations/<model_dir>/<mask_strategy>/<prompt_variant>/
├── manifest.json             # tensor shape, persona names, sample ids
└── <persona_id>.safetensors
```

`<model_dir>` is the model name with `/` replaced by `__`.

The manifest stores compact sample ids (`qa.qid`) instead of full question text,
plus tensor shape fields used for validation. Each safetensors file contains a
single `activations` tensor with shape `(num_layers, hidden_size)`.

## CLI

`extract`, `analyze`, and `steer` are implemented.

```bash
# Extract activations
# Defaults to all supported variants: templated and biography.
python main.py extract --model google/gemma-2-2b-it

# Extract only the Assistant baseline
python main.py extract --model google/gemma-2-2b-it --persona-id baseline_assistant

# Re-run personas already present in the local manifest
python main.py extract --model google/gemma-2-2b-it --persona-id baseline_assistant --force

# Run remotely on NDIF. If the remote fast path OOMs, extraction automatically
# retries that persona/variant with layer-chunked traces.
python main.py extract --model google/gemma-2-9b-it --backend remote

# Analyze saved activations
python main.py analyze --model google/gemma-2-9b-it --variant biography --mask-strategy answer_mean --out ./plots

# Run steering (example)
python main.py steer --layer 10 --model "google/gemma-2-9b-it" --persona-id 005e1868-4e17-47e3-94fa-0d20e8d93662
```

## Publishing to the Hugging Face Hub

Saved activations can be packaged as a Hugging Face dataset and pushed to the
Hub. Each `(model, mask_strategy)` pair is a dataset config, and each prompt
variant is a split. Each row is one persona with a
`(num_layers, hidden_size)` vector.

```bash
# One-time: huggingface-cli login (or set HF_TOKEN in .env)
uv run python main.py push \
    --model google/gemma-2-9b-it \
    --repo implicit-personalization/synth-persona-vectors
```

Adding more personas later: re-run `extract` (it skips personas already in the
local manifest unless `--force` is passed), then re-run `main.py push`.
Python callers can use `persona_vectors.hub.push_to_hub(...)` directly.

`scripts/extraction.sh` extracts `baseline_assistant` plus the first `N`
personas in one batch, then pushes to the Hub:

```bash
MODEL=google/gemma-2-9b-it N=100 BACKEND=remote VARIANT=templated scripts/extraction.sh
```

### Loading an existing Hub dataset

```python
from persona_vectors.artifacts import HFActivationStore

store = HFActivationStore(
    "implicit-personalization/synth-persona-vectors",
    "google/gemma-2-9b-it",
    mask_strategy="answer_mean",
)

available_variants = store.available_variants(["biography", "templated"])
variant = available_variants[0]
vectors = store.load(variant, "<UUID>")
persona_ids = store.list_personas([variant])
```

`HFActivationStore` is read-only, but exposes the same core methods as the
local `ActivationStore`: `load`, `available_variants`, `list_personas`, and
`persona_names`.
Request variants in preference order when the published dataset does not have
every local prompt variant yet.
