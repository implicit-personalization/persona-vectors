# Persona Vectors

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
│   ├── notebook_extract.py      # Extract activations from model (miniaml PoC)
│   └── notebook_load.py         # Load saved activations and recreate plots
├── src/
│   ├── activation_io.py        # Save/load activations
│   ├── activations.py          # Core: extract_activations / masked_mean
│   ├── synth_persona_io.py     # SynthPersona dataset loader
│   ├── prompt_format.py        # Chat template formatting
│   ├── plots.py                # Layer-wise similarity plots (Plotly)
│   └── environment.py          # Seed and environment helpers
├── artifacts/                  # Saved activations (gitignored)
└── main.py                     # CLI entry point (WIP)
```

## Installation

```bash
uv sync
cp .env.example .env
```

## Quickstart

```bash
# Extract activations (run this first)
uv run python -m notebooks.notebook_extract

# Load saved activations / analyze
uv run python -m notebooks.notebook_load
```

## How It Works

### Two Notebooks

`notebook_extract.py` runs the full flow end to end:

1. Load dataset questions and answers
2. Extract per-question activations
3. Save them to disk
4. Mask and average the selected token spans

`notebook_load.py` is the simpler "load from disk and analyze" example.

### Saved Format

Each extraction produces:

```
artifacts/activations/<model_dir>/<prompt_variant>/<persona_id>/
├── activations.safetensors   # Per-question hidden states
└── metadata.json            # Dataset qids in saved order
```

At load time, `response_start_idx` is reconstructed by:

1. Looking up questions/answers from the dataset via saved qids
2. Formatting the prompt (system + question + answer)
3. Tokenizing to find the boundary

## CLI (WIP)

> The idea is to support something like this

```bash
# Extract activations
python main.py extract --model google/gemma-2-2b-it --out ./activations

# Analyze saved activations
python main.py analyze --activations ./activations --out ./plots --similarity cosine

# Run steering (example)
python main.py steer --layer 10 --model "google/gemma-2-9b-it" --persona-id 005e1868-4e17-47e3-94fa-0d20e8d93662
```
