# persona-vectors

Extract persona vectors from language models, then compare those vectors across layers and prompt variants, probe them for attribute information, or use them for experimental steering.

> This project is experimental.

## Flow

```text
personas + QA pairs -> prompts -> token masks -> hidden states -> saved vectors -> analysis
```

Extraction saves one `(num_layers, hidden_size)` persona vector per persona, prompt variant, model, and mask strategy. Downstream code loads those tensors for PCA, UMAP, Isomap, centered cosine similarity, clustering, linear probes, and experimental steering.

## Install

```bash
uv sync
cp .env.example .env
```

Set `NDIF_API_KEY` to use remote extraction.

## Common Commands

```bash
uv run python main.py extract --model google/gemma-2-9b-it --backend remote
uv run python main.py analyze --model google/gemma-2-9b-it --variant biography
uv run python main.py probe --model google/gemma-2-9b-it --variant templated
uv run python main.py steer --model google/gemma-2-9b-it --persona-id <UUID> --layer 20
```

## Reference

| Page | Contents |
| --- | --- |
| [Activation Extraction](extraction.md) | prompt formatting, masks, and NDIF extraction |
| [Artifacts](artifacts.md) | local `PersonaVectorStore` + Hub `HFPersonaVectorStore` |
| [Analysis](analysis.md) | vector loading, similarity, PCA, UMAP, Isomap, clustering, plots |
| [Probes](probes.md) | linear probes over saved persona vectors |
| [Steering](steering.md) | biography-minus-templated steering vectors |
