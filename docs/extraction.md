# Activation Extraction

Extraction formats persona QA samples, builds token masks, runs the model, and saves one mean activation tensor per persona and prompt variant.

Core modules:

- `src/persona_vectors/extraction.py`
- `src/persona_vectors/activations.py`
- `src/persona_vectors/preview.py`

## CLI

```bash
# All personas, both prompt variants
uv run python main.py extract --model google/gemma-2-9b-it

# One variant or selected personas
uv run python main.py extract --model google/gemma-2-9b-it --variants biography
uv run python main.py extract --model google/gemma-2-9b-it --persona-id <UUID> baseline_assistant

# First N personas, remote backend, or forced re-run
uv run python main.py extract --model google/gemma-2-9b-it --sample-size 100
uv run python main.py extract --model google/gemma-2-9b-it --backend remote
uv run python main.py extract --model google/gemma-2-9b-it --persona-id <UUID> --force
```

`extract` skips personas already present in the local manifest unless `--force` is passed. Use `--verbose` to preview token masks.

## API

```python
from persona_vectors.extraction import MaskStrategy, run_extraction

results = run_extraction(
    model=model,
    model_name="google/gemma-2-9b-it",
    qa_pairs=qa_pairs,
    variants=("templated", "biography"),
    persona=persona,
    mask_strategy=MaskStrategy.ANSWER_MEAN,
    remote=True,
)
```

Low-level extraction is also available:

```python
from persona_vectors.activations import extract_activations

vectors = extract_activations(
    model,
    input_ids_list=[input_ids],
    token_masks=[token_mask],
    remote=False,
)
```

`extract_activations()` returns `(num_layers, hidden_size)`, averaged across samples and selected tokens.

## Mask Strategies

| Strategy | Tokens averaged |
| --- | --- |
| `PERSONA_MEAN` | all persona/system-prompt tokens |
| `PERSONA_LAST` | last persona/system-prompt token |
| `QUESTION_LAST` | last user-question token |
| `QUESTION_LAST_SPECIAL` | special token immediately after the question |
| `ANSWER_PREVIOUS` | token before the first answer token |
| `ANSWER_FIRST` | first answer token |
| `ANSWER_LAST` | last answer token |
| `ANSWER_MEAN` | all answer tokens |

The default is `ANSWER_MEAN`. Persona masks use only the rendered system prompt, so extraction only needs one QA pair for those strategies.

`ANSWER_PREVIOUS` probes the position that predicts the first answer token. `QUESTION_LAST_SPECIAL` is only valid when the tokenizer renders a special delimiter immediately after the question span; otherwise extraction raises.

## Prepared Inputs

`prepare_inputs()` returns `PreparedInput` objects with:

- `input_ids`: tokenized chat prompt
- `token_mask`: boolean mask aligned to `input_ids`
- `spans`: character and token spans for template, question, and response
- `sample_id`, `question`, `prompt_text`, `offset_mapping`

The prompt is tokenized with `add_special_tokens=False` because the chat template already includes the model's special tokens.

## Remote Extraction

Set `NDIF_API_KEY` and pass `--backend remote` or `remote=True`. If NDIF raises an out-of-memory error on the fast path, extraction retries with layer-chunked traces using `model.skip_layers`. This reduces peak memory at the cost of more NDIF round trips. Local out-of-memory errors are not retried automatically.
