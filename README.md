# Persona Vectors

[![Docs](https://img.shields.io/badge/docs-view-purple?logo=github)](https://implicit-personalization.github.io/persona-vectors/)
[![PyPI](https://img.shields.io/pypi/v/persona-vectors?logo=pypi&label=PyPI)](https://pypi.org/project/persona-vectors/)

Extract persona-aligned activation vectors from language models and analyze how persona prompts move hidden states.

> This project is experimental.

## Install

```bash
uv sync
cp .env.example .env
```

Python `>=3.12` is required. Set `NDIF_API_KEY` in `.env` to run extraction remotely on NDIF.

Dataset loading comes from the sibling [`persona-data`](../persona-data) package. For local development, uncomment the `persona-data` path source in `pyproject.toml` and keep that repo checked out next to this one.

The Streamlit UI lives in the sibling [`persona-ui`](../persona-ui) repo.

## Quickstart

```bash
# Extract activations
uv run python main.py extract --model google/gemma-2-9b-it --backend remote

# Analyze saved activations
uv run python main.py analyze --model google/gemma-2-9b-it --variant biography --mask-strategy answer_mean

# Compute an experimental steering vector
uv run python main.py steer --model google/gemma-2-9b-it --persona-id <UUID> --layer 20
```

The notebooks are useful for exploratory runs:

```bash
uv run python -m notebooks.notebook_extract
uv run python -m notebooks.notebook_manifold
uv run python -m notebooks.notebook_similarity
uv run python -m notebooks.notebook_steer
```

## What Gets Saved

Extraction writes one `(num_layers, hidden_size)` tensor per persona, prompt variant, model, and mask strategy:

```text
artifacts/activations/<model_dir>/<mask_strategy>/<prompt_variant>/
├── manifest.json
└── <persona_id>.safetensors
```

`<model_dir>` is the model name with `/` replaced by `__`. Each safetensors file contains one `activations` tensor. The manifest stores tensor shape, persona names, and contributing QA sample ids.

## CLI

```bash
# Extract all personas and both prompt variants
uv run python main.py extract --model google/gemma-2-9b-it

# Extract specific personas with a train split cap
uv run python main.py extract --model google/gemma-2-9b-it --persona-id <UUID> baseline_assistant --n-train 50

# Extract the first N personas from the dataset
uv run python main.py extract --model google/gemma-2-9b-it --sample-size 100

# Re-run personas already present locally
uv run python main.py extract --model google/gemma-2-9b-it --persona-id <UUID> --force

# Push local activations to the Hub
uv run python main.py push --model google/gemma-2-9b-it --repo implicit-personalization/synth-persona-vectors
```

See the [docs](https://implicit-personalization.github.io/persona-vectors/) for API details.

## Layout

```text
src/persona_vectors/
├── activations.py   # low-level hidden-state extraction
├── extraction.py    # prompt formatting, masks, persona extraction flow
├── artifacts.py     # local and Hub activation stores
├── analysis.py      # loading, PCA, cosine similarity, clustering
├── plots.py         # Plotly figures
├── steering.py      # experimental steering vectors
└── parser.py        # CLI parser
```
