# Persona Vectors

Extract persona-aligned activation vectors from language models and experiment with activation steering.

> [!WARNING]
> There is some code currently fully AI generated that just serves as testing but can safly be removed in the future
> I (Jacopo) left comment on most of the files to explain what I think should be changed and worked on

> Anybody feel free to change priority in the tasks and change things (this is very experimental currently 🚨)

## Overview

Given a set of personas and evaluation questions, this project:

1. Formats each persona as a system prompt (short templated or long biography)
2. Generates model responses to each question under that persona (perhaps given as input already)
3. Extracts hidden states over the response tokens at every layer
4. Averages those hidden states across questions to produce a **persona vector** per layer

The resulting vectors can be compared across layers (cosine similarity, Euclidean distance) and eventually used for steering experiments.

## Repository Layout

```
persona-vectors/
├── notebook.py                # Exploration script: loads model, extracts and compares activations
├── main.py                    # CLI entry point (extract / analyze subcommands — WIP)
├── pyproject.toml             # Dependencies (managed with uv)
├── docs/
│   └── plan.md                # Working plan and TODOs
└── src/
    ├── activations.py         # extract_activations / get_mean_activations
    ├── format.py              # Chat template formatting + response_start_idx computation
    ├── load.py                # Persona JSONL loading (load_personas, get_persona_name)
    ├── plots.py               # Layer-wise similarity plots (Plotly)
    └── environment.py         # Seed and environment helpers
```

## Installation

This project uses [uv](https://github.com/astral-sh/uv) and `pyproject.toml`.

```bash
uv sync
```

## Quickstart

> I have set it up like this to run it cell by cell via the REPL but it can be also changed to be a jupyter or better a [marimo notebook](https://marimo.io/)

Run the exploration notebook to load a model, extract persona vectors, and compare them across layers:

```bash
python notebook.py
```

This will:

- Load `google/gemma-2-2b-it` (change `MODEL_NAME` at the top of the file)
- Load personas from `dataset_personas.jsonl`
- Generate responses to a small set of evaluation questions under two prompt variants (`templated_prompt` vs `biography_md`)
- Compute mean hidden states per layer for each variant

> **Note:** Response generation is done locally via `nnsight`.
> Activations are averaged over response tokens only — the system prompt and question tokens are excluded .

## Core Concepts

### Prompt Variants

Each persona in the dataset ships with two system prompt variants:

| Field              | Description                                          |
| ------------------ | ---------------------------------------------------- |
| `templated_prompt` | Short structured prompt (e.g. name, age, occupation) |
| `biography_md`     | Long narrative biography for the sasme persona       |

## CLI (WIP)

> [!WARNING]
> This is just a basic template for later to work on with the functionality I thought might be useful

`main.py` exposes two subcommands that are scaffolded but not yet implemented:

```bash
# Extract activations
python main.py extract --model <model_name_or_path>  --input dataset_personas.jsonl --out ./activations

# Analyze saved activations
python main.py analyze --activations ./activations --out ./plots --similarity cosine
```

## Roadmap

- [ ] Conclude work on the notebook to actually have extraction of the basic persona vectors -> And extend it to real word (with support for question/answers from William's work)
- [ ] Load pre-generated responses from SynthPersona / PersonaGuess instead of regenerating
- [ ] Wire up `main.py` extraction and analysis to use `src/` functions
- [ ] Support passing different JSONL files and persona prompt formats via CLI
- [ ] Scale to the full question set (and batch with vLLM)
- [ ] Save persona vectors to disk (`.pt` / `.safetensors` or whatever is best), -> upload to Hugging Face potentially
- [ ] Exploratory analyses with plots and similarity metrics
- [ ] Steering experiments using the extracted persona vectors
