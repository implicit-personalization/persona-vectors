# Activation Extraction

Extract hidden states from a model and average them over selected tokens and QA pairs.
Use `extract_activations()` for the low-level primitive and `run_extraction()` for the full persona flow.
Core modules: `src/persona_vectors/activations.py`, `src/persona_vectors/extraction.py`, and `src/persona_vectors/preview.py`.

## Quick Start

```python
import torch

from persona_vectors.activations import extract_activations

activations = extract_activations(
    model,
    input_ids_list=[torch.tensor([1, 2, 3, 4])],
    token_masks=[torch.tensor([False, False, True, True])],
    remote=False,
)
```

## CLI

```bash
# Extract all supported prompt variants: templated and biography.
uv run python main.py extract --model google/gemma-2-9b-it

# Extract one prompt variant.
uv run python main.py extract --model google/gemma-2-9b-it --variants biography

# Extract one persona or the Assistant baseline.
uv run python main.py extract --model google/gemma-2-9b-it --persona-id <UUID>
uv run python main.py extract --model google/gemma-2-9b-it --persona-id baseline_assistant

# Extract a set of personas in one run.
uv run python main.py extract --model google/gemma-2-9b-it --persona-id <UUID> baseline_assistant

# Extract the first N personas from the dataset.
uv run python main.py extract --model google/gemma-2-9b-it --sample-size 100

# Re-run personas already listed in the local manifest.
uv run python main.py extract --model google/gemma-2-9b-it --persona-id <UUID> --force

# Run on NDIF instead of locally.
uv run python main.py extract --model google/gemma-2-9b-it --backend remote
```

`--variants` accepts one or more prompt variants. `extract` skips personas
already present in the local manifest by default; pass `--force` to re-run
them. Use `--persona-id` to select one or more explicit personas, or
`--sample-size` to load the first N personas from the dataset. Use `--verbose`
to print token-mask previews.

## Key Functions

- `prepare_inputs()`: formats QA pairs and builds token masks, returning a list of `PreparedInput`
- `extract_activations()`: runs the forward pass and returns one `(num_layers, hidden_size)` mean tensor
- `run_extraction()`: full persona/variant flow used by the CLI and `notebook_extract.py`
- `persona_vectors.preview.preview_prepared_inputs()`: pretty-prints prepared samples with masked tokens highlighted (useful when iterating on a new `MaskStrategy`)
- `persona_vectors.preview.preview_token_segments()`: returns renderer-neutral token segments for UI previews

### `PreparedInput`

Each element returned by `prepare_inputs()` bundles a formatted sample together with everything needed to line up masks with the tokenized prompt:

- `question`, `prompt_text`: original question and fully rendered chat prompt
- `input_ids`: 1-D token ids for the prompt (no added BOS; the chat template already includes it)
- `token_mask`: boolean mask over `input_ids` — `True` values are averaged
- `spans`: token + character ranges for the `template`, `question`, and `response` segments
- `offset_mapping`: character offsets per token (used by `persona_vectors.preview`)

## Masking

`MaskStrategy` selects which tokens contribute to the averaged hidden state. The default is `ANSWER_MEAN`.

| Strategy | Token(s) averaged |
|---|---|
| `PERSONA_MEAN` | All persona/system-prompt tokens before the question |
| `PERSONA_LAST` | Last persona/system-prompt token before the question |
| `ANSWER_MEAN` | Every assistant-answer token (default) |
| `ANSWER_PREVIOUS` | Token immediately before the first assistant-answer token |
| `ANSWER_FIRST` | First assistant-answer token |
| `ANSWER_LAST` | Last assistant-answer token |
| `QUESTION_LAST` | Last token of the user question |
| `QUESTION_LAST_SPECIAL` | First special token after the question span (often a chat-template delimiter) |

`PERSONA_MEAN` and `PERSONA_LAST` refer to the persona prefix, which is the system-prompt/template portion for both templated and biography variants. They do not include the user question.

`ANSWER_PREVIOUS` is useful for probing the state that will predict the first answer token, because causal language models compute the next-token distribution from the position immediately before that token.

`QUESTION_LAST_SPECIAL` raises if the token immediately after the question span is not a tokenizer special id, so it only makes sense for chat templates that end each turn with a delimiter token.

## Note

For NDIF runs, pass `remote=True` to `extract_activations()` and set `NDIF_API_KEY` in your environment.

To keep multiple extraction runs separate, pass `activations_dir` to `run_extraction()`:

```python
run_extraction(..., activations_dir="artifacts/activations/run_001")
```

`run_extraction()` saves one activation tensor per persona and prompt variant.
The tensor has shape `(num_layers, hidden_size)`: each layer vector is already
averaged over all prepared QA pairs and over the tokens selected by the mask
strategy. The manifest still records the contributing QA `sample_ids` for
provenance.

### Long biographies / OOM

Remote NDIF extraction first tries the fast path: one `model.session(...)` for
the selected persona/variant, with one `model.trace(...)` per prepared question.

If that remote fast path raises an `OutOfMemoryError`, `extract_activations()`
automatically retries the whole persona/variant with chunked extraction. The
chunk size is currently chosen as `max(1, model.num_layers // 4)`.

Chunked extraction slices each question across layer chunks and carries the
boundary residual stream forward with `model.skip_layers`. This bounds peak
memory, but it is slower because it performs one NDIF round trip per question
per layer chunk.

Local OOMs are not retried automatically, and the current CLI does not expose a
manual `--chunk-size` option.
