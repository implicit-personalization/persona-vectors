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

- `extract_activations()`: runs the forward pass and returns masked mean activations
- `prepare_inputs()`: formats QA pairs and builds token masks
- `MaskStrategy`: chooses which tokens to average
- `run_extraction()`: full persona/variant extraction flow used by the CLI

## Masking

The default strategy is `MaskStrategy.RESPONSE_MEAN`, which averages only the assistant response tokens.

Other built-in options:

- `RESPONSE_FIRST`
- `RESPONSE_LAST`
- `PROMPT_MEAN`
- `PROMPT_LAST`

`PROMPT_LAST` selects the token immediately before the assistant response, which is often a newline or template marker.

## Note

For NDIF runs, instantiate the model without `device_map` or `dtype` so it loads on the meta device.
