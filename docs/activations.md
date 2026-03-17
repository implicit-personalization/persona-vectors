# Activation Extraction

Extract hidden state activations from language models at specific token positions.
Core module: `src/activations.py`

---

This is the first step in the persona vectors pipeline:

```
Dataset → Format Prompts → Extract Activations → Save → Analyze/Compare
```

---

## Quick Start

```python
from src.activations import extract_activations

# Extract activations for response tokens only
activations = extract_activations(
    model,
    full_texts=["prompt + response text"],
    token_masks=[torch.tensor([False, ..., True])],
    remote=False,
)
# Returns: (n_texts, n_layers, d_model) tensor
```

---

## extract_activations()

Core function that runs the model forward pass and computes masked mean activations.

```python
from src.activations import extract_activations

activations = extract_activations(
    model,
    full_texts=["prompt + response text"],
    token_masks=[torch.tensor([False, ..., True])],
    remote=False,
)
```

> **WARNING (remote=True):** When using remote execution, instantiate the model
> without `device_map`/`dtype` so it loads on the meta device. OOM errors may appear
> unreliably — chat with me if you encounter issues.

---

## Creating Token Masks

The token mask tells the extraction function which positions to include when computing
the mean activation. Typically, you want **response tokens only** (not the prompt).

### Using format_messages()

Use `prompt_format.format_messages()` to get the response start index, then build
a mask for response tokens only:

```python
from src.prompt_format import format_messages

full_prompt, response_start_idx = format_messages(messages, tokenizer)
input_ids = tokenizer(full_prompt, return_tensors="pt").input_ids[0]

# Create mask: True for response tokens, False for prompt tokens
mask = torch.zeros(len(input_ids), dtype=torch.bool)
mask[response_start_idx:] = True  # Response tokens only
```

### Alternative strategies

- **Last N tokens:** `mask[-n:] = True` — useful for focusing on recent generation
- **Last Token Before response:** `mask[response_start_idx - 1] = True` - useful for analyzing what triggers a particular persona's response style
- **All tokens:** `mask[:] = True` — include everything (prompt + response)
- **Sliding window:** Extract at multiple positions for more granular analysis

---

## Remote Execution

For larger models that don't fit in local GPU memory, use NDIF remote execution.

| Scenario      | Model                 | Execution                                                  |
| ------------- | --------------------- | ---------------------------------------------------------- |
| Local testing | `gemma-2-2b-it`       | `remote=False` with `device_map="auto"`, `dtype="float16"` |
| Large models  | `gemma-2-9b-it`, etc. | `remote=True` — no local GPU needed                        |

> **WARNING:** If you get `RemoteException: RecursionError`, the NDIF server is
> running nnsight <0.6.2 with a ModuleList integer-index proxy bug. Wait for
> the server to update.

---

## Troubleshooting

| Issue                    | Solution                                                        |
| ------------------------ | --------------------------------------------------------------- |
| OOM on remote            | Check NDIF server status - try again later                      |
| RecursionError on remote | Server nnsight version is outdated                              |
| Empty activations        | Check that your token mask has True values at correct positions |
