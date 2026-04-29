# Persona Vectors

[![Docs](https://img.shields.io/badge/docs-view-purple?logo=github)](https://implicit-personalization.github.io/persona-vectors/)

Extract persona-aligned activation vectors from language models and analyze how persona prompts move hidden states.

> [!WARNING]
> This is very experimental currently 🚨

## Overview

Given a set of personas and evaluation questions, this project:

1. Formats each persona as a system prompt (short `templated` or long `biography`)
2. Extracts hidden states at each layer with configurable token masking
3. Saves per-question, per-layer hidden states, then averages them into persona-level views for analysis

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
│   ├── plots.py                 # Plotly figures for layer-wise analysis
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

# Build interactive persona-mean PCA and similarity plots from saved activations
uv run python main.py analyze --model google/gemma-2-9b-it --variant biography --mask-strategy answer_mean

# Compute a steering vector from saved activations
uv run python main.py steer --persona-id <UUID> --model google/gemma-2-9b-it --layer 20
```

## Streamlit App

The Streamlit UI lives in the sibling [persona-ui](../persona-ui) repo.

## How It Works

### Notebooks

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
artifacts/activations/<model_dir>/<mask_strategy>/<prompt_variant>/
├── manifest.json             # tensor shape, persona names, sample ids
└── <persona_id>.safetensors
```

`<model_dir>` is the model name with `/` replaced by `__`.

The manifest stores compact sample ids (`qa.qid`) instead of full question text,
plus tensor shape fields used for validation.

The Assistant baseline is exposed as a regular variant (`baseline`) in the
extraction CLI and UI. It is persona-less, so it is run once across the first
selected persona's QA pairs and stored under the shared baseline persona id.
Compare views can add it as an Assistant reference alongside templated or
biography persona samples.

## CLI

`extract`, `analyze`, and `steer` are implemented.

```bash
# Extract activations (defaults to all variants, including baseline)
python main.py extract --model google/gemma-2-2b-it

# Pick specific variants — 'baseline' is just another variant and is run once
python main.py extract --model google/gemma-2-2b-it --variants biography baseline

# Analyze saved activations
python main.py analyze --model google/gemma-2-9b-it --variant biography --mask-strategy answer_mean --out ./plots

# Run steering (example)
python main.py steer --layer 10 --model "google/gemma-2-9b-it" --persona-id 005e1868-4e17-47e3-94fa-0d20e8d93662

# Load steering activations extracted with a non-default mask strategy
python main.py steer --layer 10 --model "google/gemma-2-9b-it" --persona-id <UUID> --mask-strategy answer_previous
```
