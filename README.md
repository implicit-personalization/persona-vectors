# Persona Vectors

[![Docs](https://img.shields.io/badge/docs-view-purple?logo=github)](https://implicit-personalization.github.io/persona-vectors/)
[![PyPI](https://img.shields.io/pypi/v/persona-vectors?logo=pypi&label=PyPI)](https://pypi.org/project/persona-vectors/)

Extract persona vectors from language models, then probe, project, or steer with them.

A *persona vector* is the mean hidden-state activation a model produces while answering as a given persona. Extraction saves one `(num_layers, hidden_size)` tensor per persona, prompt variant, model, and mask strategy; everything downstream reads those tensors back.

```text
personas + QA pairs -> prompts -> token masks -> hidden states -> saved vectors -> analysis
```

## Install

```bash
uv sync
cp .env.example .env
```

Requires Python `>=3.12`. Set `NDIF_API_KEY` in `.env` for remote extraction.

Dataset loading comes from [`persona-data`](https://github.com/implicit-personalization/persona-data); the

## Quickstart

```bash
# Extract — one (num_layers, hidden_size) vector per persona/variant/mask
uv run python main.py extract --model google/gemma-2-9b-it --backend remote

# Analyze — PCA, similarity, clustering, scree plots
uv run python main.py analyze --model google/gemma-2-9b-it --variant biography

# Probe — linear probes per persona attribute
uv run python main.py probe --model google/gemma-2-9b-it --variant templated

# Steer — biography minus templated direction
uv run python main.py steer --model google/gemma-2-9b-it --persona-id <UUID> --layer 20

# Push extracted vectors to the Hub
uv run python main.py push --model google/gemma-2-9b-it --repo implicit-personalization/synth-persona-vectors
```

Notebooks under `notebooks/` cover the same flows interactively.

## Extraction scripts

```bash
# Steering: train split, push to Hub, refresh the dataset card
MODEL=google/gemma-2-9b-it scripts/extraction_train_split.sh

# All-questions (explicit only): first 100 personas under artifacts/persona-vectors/,
# then push and refresh the dataset card
MODEL=google/gemma-2-9b-it scripts/extraction_all_questions.sh
```

## What gets saved

```text
artifacts/activations/<model_dir>/<mask_strategy>/<prompt_variant>/
├── manifest.json
└── <persona_id>.safetensors
```

`<model_dir>` is the HF id with `/` → `__`. Each safetensors file holds one `activations` tensor — the persona vector for that variant, averaged across QA pairs and selected tokens. 
`scripts/extraction_all_questions.sh` writes under `artifacts/persona-vectors/` to keep all-questions runs separate from train-split runs; pass `--activations-dir artifacts/persona-vectors` to read it back. 
See the [artifacts docs](https://implicit-personalization.github.io/persona-vectors/artifacts/) for the full layout.

## Layout

```text
src/persona_vectors/
├── activations.py   # low-level hidden-state extraction
├── extraction.py    # prompt formatting, masks, persona extraction flow
├── preview.py       # token-mask preview for --verbose
├── artifacts.py     # PersonaVectorStore (local) + HFPersonaVectorStore (Hub)
├── hub.py           # push to / discover Hub vector datasets
├── analysis.py      # aligned dataset loading, PCA, cosine similarity, clustering
├── plots/           # Plotly figures
├── attributes.py    # attribute schema + color helpers for plots
├── probes.py        # linear probes over saved persona vectors
├── steering.py      # experimental steering vectors
└── parser.py        # CLI parser
```

See the [docs](https://implicit-personalization.github.io/persona-vectors/) for API details.
