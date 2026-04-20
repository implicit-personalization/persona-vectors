# Activation Extraction

Extract hidden states from a model and average them over selected tokens.
Use `extract_activations()` for the low-level primitive and `run_extraction()` for the full persona flow.
Core modules: `src/persona_vectors/activations.py` and `src/persona_vectors/extraction.py`

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

## Key Functions

- `prepare_inputs()`: formats QA pairs and builds token masks, returning a list of `PreparedInput`
- `extract_activations()`: runs the forward pass and returns the masked-mean hidden states
- `run_extraction()`: full persona/variant flow used by the CLI and `notebook_extract.py`
- `preview_prepared_inputs()`: pretty-prints prepared samples with masked tokens highlighted (useful when iterating on a new `MaskStrategy`)

### `PreparedInput`

Each element returned by `prepare_inputs()` bundles a formatted sample together with everything needed to line up masks with the tokenized prompt:

- `question`, `prompt_text`: original question and fully rendered chat prompt
- `input_ids`: 1-D token ids for the prompt (no added BOS; the chat template already includes it)
- `token_mask`: boolean mask over `input_ids` — `True` values are averaged
- `spans`: token + character ranges for the `template`, `question`, and `response` segments
- `offset_mapping`: character offsets per token (used by the preview renderer)

## Masking

`MaskStrategy` selects which tokens contribute to the averaged hidden state. The default is `ANSWER_MEAN`.

| Strategy | Token(s) averaged |
|---|---|
| `ANSWER_MEAN` | Every assistant-answer token (default) |
| `ANSWER_FIRST` | First assistant-answer token |
| `ANSWER_LAST` | Last assistant-answer token |
| `QUESTION_LAST` | Last token of the user question |
| `QUESTION_LAST_SPECIAL` | First special token after the question span (often a chat-template delimiter) |

`QUESTION_LAST_SPECIAL` raises if the token immediately after the question span is not a tokenizer special id, so it only makes sense for chat templates that end each turn with a delimiter token.

## Note

For NDIF runs, instantiate the model without `device_map` or `dtype` so it loads on the meta device.
