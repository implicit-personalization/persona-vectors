# Persona Vectors

[![Docs](https://img.shields.io/badge/docs-view-purple?logo=github)](https://implicit-personalization.github.io/persona-vectors/)
[![PyPI](https://img.shields.io/pypi/v/persona-vectors?logo=pypi&label=PyPI)](https://pypi.org/project/persona-vectors/)

Extract persona vectors from language models and analyze how persona prompts move hidden states. The same vectors feed linear probes and experimental steering.

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
# Extract persona vectors
uv run python main.py extract --model google/gemma-2-9b-it --backend remote

# Analyze saved vectors
uv run python main.py analyze --model google/gemma-2-9b-it --variant biography --mask-strategy answer_mean

# Train linear probes over saved vectors
uv run python main.py probe --model google/gemma-2-9b-it --variant templated

# Compute an experimental steering vector
uv run python main.py steer --model google/gemma-2-9b-it --persona-id <UUID> --layer 20
```

The notebooks are useful for exploratory runs:

```bash
uv run python -m notebooks.notebook_extract
uv run python -m notebooks.unsupervised.manifold
uv run python -m notebooks.unsupervised.similarity
uv run python -m notebooks.notebook_steer

# Per-task probe notebooks
uv run python -m notebooks.probes.binary
uv run python -m notebooks.probes.categorical
uv run python -m notebooks.probes.ordinal
uv run python -m notebooks.probes.numeric
```

## Extraction Scripts

```bash
# Persona vectors for steering: train split, push to the Hub
MODEL=google/gemma-2-9b-it scripts/extraction_train_split.sh

# All-questions workflow (explicit only): first 100 personas, save under
# artifacts/persona-vectors, then push to the Hub
MODEL=google/gemma-2-9b-it scripts/extraction_all_questions.sh
```

The extraction scripts refresh the Hugging Face dataset card (`README.md`) after pushing vectors.

## What Gets Saved

Extraction writes one `(num_layers, hidden_size)` persona vector per persona, prompt variant, model, and mask strategy:

```text
artifacts/activations/<model_dir>/<mask_strategy>/<prompt_variant>/
├── manifest.json
└── <persona_id>.safetensors
```

`<model_dir>` is the model name with `/` replaced by `__`. Each safetensors file contains one `activations` tensor — the persona vector for that variant, averaged across QA pairs and selected tokens. The manifest stores tensor shape, persona names, and contributing QA sample ids.

`scripts/extraction_all_questions.sh` writes under `artifacts/persona-vectors/` instead, so you can keep all-questions runs separate from train-split runs. Pass `--activations-dir artifacts/persona-vectors` to subsequent `analyze` / `probe` / `steer` calls to read them back.

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

# Push local persona vectors to the Hub
uv run python main.py push --model google/gemma-2-9b-it --repo implicit-personalization/synth-persona-vectors
```

See the [docs](https://implicit-personalization.github.io/persona-vectors/) for API details.

## Layout

```text
src/persona_vectors/
├── activations.py   # low-level hidden-state extraction
├── extraction.py    # prompt formatting, masks, persona extraction flow
├── artifacts.py     # PersonaVectorStore (local) + HFPersonaVectorStore (Hub)
├── analysis.py      # loading, PCA, cosine similarity, clustering
├── plots/           # Plotly figures; plots.probes hosts probe-specific views
├── probes.py        # linear probes over saved persona vectors
├── steering.py      # experimental steering vectors
└── parser.py        # CLI parser
```
