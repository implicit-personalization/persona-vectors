# persona-vectors

Extract persona vectors from language models, then compare those vectors across
layers and prompt variants, probe them for attribute information, or use them for experimental steering.

> This project is experimental.

## What is a persona vector?

A persona vector is the mean hidden-state activation a model produces while
answering as a given persona. Extraction saves one `(num_layers, hidden_size)`
tensor per persona, prompt variant, model, and mask strategy. Every downstream tool — similarity, projection, probes, steering — reads those saved tensors back; nothing re-runs the model.

## Pipeline

```text
personas + QA pairs -> prompts -> token masks -> hidden states -> saved vectors -> analysis
```

| Stage | What happens | Reference |
| --- | --- | --- |
| Extraction | Format persona QA prompts, build token masks, run the model, save one vector per persona and prompt variant | [Activation Extraction](extraction.md) |
| Storage | Local `PersonaVectorStore` and read-only Hub `HFPersonaVectorStore` over one shared on-disk layout | [Artifacts](artifacts.md) |
| Analysis | Aligned vector loading, centered cosine similarity, PCA / UMAP / Isomap, clustering, plots | [Analysis](analysis.md) |
| Probes | Linear probes that read a persona attribute out of the vectors | [Probes](probes.md) |
| Steering | Experimental biography-minus-templated direction | [Steering](steering.md) |

## Install

```bash
uv sync
cp .env.example .env
```

Requires Python `>=3.12`. Set `NDIF_API_KEY` to use remote extraction. See the [README](https://github.com/implicit-personalization/persona-vectors#readme) for quickstart commands and extraction scripts.
