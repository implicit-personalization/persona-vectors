# Activation Extraction

Core module: `src/activations.py`

---

## extract_activations()

```python
from src.activations import extract_activations

activations = extract_activations(
    model,
    full_texts=["prompt + response text"],
    token_masks=[torch.tensor([False, ..., True])],
    remote=False,
)
```

**Parameters:**

- `model`: The nnsight LanguageModel.
- `full_texts`: List of full formatted prompt+response strings, one per sample.
- `token_masks`: List of boolean masks over the full (unpadded) token sequence.
- `remote`: If `True`, execute on NDIF's remote servers. Requires
  `NDIF_API_KEY` to be set. When using `remote=True`, instantiate the model
  without `device_map`/`dtype` so it loads on the meta device (note OOM might appear unreliably chat with me for any problems).

**Returns:** `torch.Tensor` of shape `(n_texts, n_layers, d_model)`.

---

## Creating Token Masks

> Example of possible mask / alternative masking eg. last token might be also a viable strategy

Use `prompt_format.format_messages()` to get the response start index, then build
a mask for response tokens only:

```python
from src.prompt_format import format_messages

full_prompt, response_start_idx = format_messages(messages, tokenizer)
input_ids = tokenizer(full_prompt, return_tensors="pt").input_ids[0]

mask = torch.zeros(len(input_ids), dtype=torch.bool)
mask[response_start_idx:] = True  # Response tokens only
```

---

## Remote Execution

- **Local:** Use smaller models (e.g., `gemma-2-2b-it`) with `dtype` and
  `device_map="auto"`.
- **Remote:** Use larger models (e.g., `gemma-2-9b-it`) on NDIF servers. No
  local GPU needed — model loads on meta device.

> **WARNING:** If you get `RemoteException: RecursionError`, the NDIF server is
> running nnsight <0.6.2 with a ModuleList integer-index proxy bug. Wait for
> the server to update.
