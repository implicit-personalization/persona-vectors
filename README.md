# Persona Vectors

[![Docs](https://img.shields.io/badge/docs-view-purple?logo=github)](https://implicit-personalization.github.io/persona-vectors/)

Extract persona-aligned activation vectors from language models and experiment with activation steering.

> [!WARNING]
> This is very experimental currently 🚨

## Overview

Given a set of personas and evaluation questions, this project:

1. Formats each persona as a system prompt (short `templated` or long `biography`)
2. Extracts hidden states at each layer (with support to then mask some specific tokens)
3. Averages those hidden states across questions to produce a **persona vector** per layer

The resulting vectors can be compared across layers (cosine similarity) and eventually used for steering experiments.

## Repository Layout

```
persona-vectors/
├── notebooks/
│   ├── notebook_extract.py      # Extraction pipeline (primary working script)
│   ├── notebook_compare.py      # Load saved activations and compare variants
│   └── notebook_steer.py        # Steering experiments
├── src/persona_vectors/
│   ├── activations.py           # Core extraction helpers
│   ├── analysis.py              # PCA / UMAP projections and scatter plots
│   ├── artifacts.py             # Save/load/query activation artifact helpers
│   ├── plots.py                 # Layer-wise cosine similarity plots
│   ├── steering.py              # Steering vector computation and application
│   └── parser.py                # CLI argument parsing
├── artifacts/                   # Saved activations (gitignored)
├── docs/                        # Reference documentation
└── main.py                      # CLI entry point
```

Dataset loading (`SynthPersonaDataset`, `PersonaGuessDataset`) and environment
helpers come from the sibling [persona-data](../persona-data) package.

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

# Load saved activations / compare variants
uv run python -m notebooks.notebook_compare

# Analyze saved activations (parsed, not implemented yet)
uv run python main.py analyze --out ./plots --similarity cosine

# Compute a steering vector from saved activations
uv run python main.py steer --persona-id <UUID> --model google/gemma-2-9b-it --layer 20
```

## Streamlit App

The Streamlit UI lives in the sibling [persona-ui](../persona-ui) repo.

## How It Works

### Two Notebooks

`notebook_extract.py` runs the full flow end to end:

1. Load dataset questions and answers
2. Extract per-question activations
3. Save them to disk
4. Mask and average the selected token spans

`notebook_compare.py` loads saved activations via `ActivationStore` and compares variants.

`notebook_steer.py` loads saved activations and computes a steering vector for a
selected persona.

### Saved Format

Each extraction produces:

```
artifacts/activations/<model_dir>/<prompt_variant>/<persona_id>/
├── activations.safetensors   # Per-question hidden states
└── metadata.json            # persona_id, persona_name, questions, n_questions, num_layers, hidden_size
```

`<model_dir>` is the model name with `/` replaced by `__`.

The metadata stores the question text directly, so load-time analysis no longer needs
to re-resolve qids from the dataset. It also stores tensor shape fields for validation
at load time.

## CLI

`extract` and `steer` are implemented. `analyze` is parsed but still raises
`NotImplementedError`.

```bash
# Extract activations
python main.py extract --model google/gemma-2-2b-it

# Analyze saved activations (not implemented yet)
python main.py analyze --out ./plots --similarity cosine

# Run steering (example)
python main.py steer --layer 10 --model "google/gemma-2-9b-it" --persona-id 005e1868-4e17-47e3-94fa-0d20e8d93662
```
