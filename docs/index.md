# persona-vectors

Extract persona-aligned activation vectors from language models and analyze how persona prompts move hidden states.

> **Warning:** This project is very experimental.

## How it works

Given a set of personas and evaluation questions:

1. Format each persona as a system prompt (`templated` or `biography` variant)
2. Build token masks over the answer span, the prediction position, or another chosen strategy
3. Extract hidden states at each layer over the masked tokens
4. Average those hidden states across questions → **persona vector** per layer

Vectors can then be compared across layers with PCA, UMAP, and centered cosine
similarity. Steering is kept as an experimental side path.

## Pipeline

```
Dataset → Format Prompts → Build Token Masks → Extract Activations → Save → Analyze
```

| Step | Doc |
|---|---|
| Build token masks and extract hidden states from the model | [Activation Extraction](extraction.md) |
| Save and load activation tensors | [Artifacts](artifacts.md) |
| Comparison and analysis views | [Analysis](analysis.md) |
| Experimental steering vectors | [Steering](steering.md) |

## Installation

```bash
uv sync
cp .env.example .env
```

Set `NDIF_API_KEY` in `.env` if you want to use remote execution for large models.

## Quickstart

```bash
# Extract activations (run this first)
uv run python -m notebooks.notebook_extract

# Same extraction flow with token-mask preview and a short sample run
# (set verbose=True in the notebook)

# Load saved activations and inspect comparison views
uv run python -m notebooks.notebook_compare

# Optional: compute an experimental steering vector from saved activations
uv run python main.py steer --persona-id <UUID> --model google/gemma-2-9b-it --layer 20

# Use non-default extracted activations, such as the pre-answer prediction position
uv run python main.py steer --persona-id <UUID> --model google/gemma-2-9b-it --mask-strategy answer_previous
```

## Dependencies

Dataset loading is provided by the sibling [`persona-data`](https://implicit-personalization.github.io/persona-data/) package, which pulls from:

- [implicit-personalization/synth-persona](https://huggingface.co/datasets/implicit-personalization/synth-persona) — persona profiles and QA pairs
- [implicit-personalization/persona-guess](https://huggingface.co/datasets/implicit-personalization/persona-guess) — turn-based persona games
