# Persona Vectors

Extract persona-aligned activation vectors from language models and experiment with activation steering.

> [!WARNING]
> There is some code currently fully AI generated that just serves as testing but can safly be removed in the future
> I (Jacopo) left comment on most of the files to explain what I think should be changed and worked on

> Anybody feel free to change priority in the tasks and change things (this is very experimental currently 🚨)

## Overview

Given a set of personas and a fixed neutral prompt bank, this project:

1. Runs an extraction grid over prompt format (`templated`, `biography`) and neutral prompts
2. Generates assistant responses under persona context and under a baseline assistant context
3. Extracts two activation summaries per run with NNsight:
   - response-token mean per layer
   - last-prompt-token activation per layer
4. Stacks and averages across neutral prompts to produce persona representations
5. Builds contrastive vectors by subtracting baseline representations

The resulting vectors can be compared across layers and across prompt formats (biography vs templated), then used for downstream steering experiments.

## Repository Layout

```
persona-vectors/
│
├── main.py                    # CLI entry point for extraction and analysis
├── notebook.py                # End-to-end pipeline for a extracting single persona vector
├── notebook_marimo.py         # Reactive Marimo web pipeline
│
├── src/
│   ├── activations.py         # extract_activations / get_mean_activations via NNsight
│   ├── activation_io.py       # Saves and loads per-question activation tensors to/from disk in safetensors + JSON format
│   ├── environment.py         # Shared utilities for loading .env + seed settings
│   ├── persona_io.py          # Loads personas and Q&A pairs from a JSONL dataset into typed Python dataclasses
│   ├── plots.py               # Computes + plots layer-wise cosine similarity between two activation tensors
│   └── prompt_format.py       # Chat template formatting + response_start_idx computation
│
├── docs/
│   └── plan.md                # Working plan and TODOs
│
├── data/
│   ├── dataset_personas.jsonl
│   ├── dataset_qa.jsonl
│   └── neutral_prompts.jsonl  # Fixed neutral prompt bank used for extraction
├── pyproject.toml             # Project dependencies (torch, nnsight, safetensors, marimo, etc.) managed with uv
├── uv.lock                    # Locked dependency versions for reproducible installs
├── .env.example               # Required environment variables (NDIF_API_KEY, HF_HOME, PERSONAS_PATH, ARTIFACTS_DIR)
├── .gitignore            
└── README.md                
```

## Installation

This project uses [uv](https://github.com/astral-sh/uv) and `pyproject.toml`.

```bash
uv sync
```

## Configuration

Copy `.env.example` to `.env` and fill in the values:

```bash
cp .env.example .env
```

## Quickstart

> I have set it up like this to run it cell by cell via the REPL but it can be also changed to be a jupyter or better a [marimo notebook](https://marimo.io/)

Run extraction from CLI:

```bash
python main.py extract \
  --model google/gemma-2-2b-it \
  --input data/dataset_personas.jsonl \
  --neutral-prompts data/neutral_prompts.jsonl \
  --out artifacts/activations
```

Run analysis after extraction:

```bash
python main.py analyze \
  --activations artifacts/activations \
  --model google/gemma-2-2b-it \
  --input data/dataset_personas.jsonl
```

This produces:
- per-neutral-prompt activation stacks `(n_prompts, n_layers, d_model)`
- averaged persona representations `(n_layers, d_model)`
- contrastive vectors `(n_layers, d_model)` vs baseline assistant
- biography vs templated layerwise cosine plots and PCA projection artifacts

## Core Concepts

### Prompt Variants

Each persona in the dataset ships with two system prompt variants:

| Field              | Description                                          |
| ------------------ | ---------------------------------------------------- |
| `templated_prompt` | Short structured prompt (e.g. name, age, occupation) |
| `biography_md`     | Long narrative biography for the sasme persona       |

## CLI

```bash
# Extract full persona grid with NNsight
python main.py extract \
  --model <model_name_or_path> \
  --input data/dataset_personas.jsonl \
  --neutral-prompts data/neutral_prompts.jsonl \
  --out artifacts/activations

# Analyze saved contrastive vectors (biography vs templated)
python main.py analyze \
  --activations artifacts/activations \
  --model <model_name_or_path> \
  --input data/dataset_personas.jsonl
```

Useful `extract` flags:
- `--remote` to execute traces on NDIF
- `--baseline-system-prompt "You are a helpful assistant."`
- `--max-new-tokens <int>`
- `--do-sample` to sample during generation
- `--persona-limit <int>` for smoke tests

## Roadmap

> Review this repo https://github.com/Butanium/assistant-axis and think of what we need from that

- [ ] Conclude work on the notebook to actually have extraction of the basic persona vectors -> And extend it to real word (with support for question/answers from William's work)
- [ ] Load pre-generated responses from SynthPersona / PersonaGuess instead of regenerating
- [ ] Wire up `main.py` extraction and analysis to use `src/` functions
- [ ] Support passing different JSONL files and persona prompt formats via CLI
- [ ] Scale to the full question set
- [x] Save persona vectors to disk (`.pt` / `.safetensors` or whatever is best)
- [ ] Exploratory analyses with plots and similarity metrics
- [ ] Steering experiments using the extracted persona vectors

### Additional notes

I would be interested in using VLLM but I don't have a nvidia GPU moreover I'm not sure about the current support of VLLM with ndif.
Moreover it would be very nice to have all of this working with nnterp possibly instead.
