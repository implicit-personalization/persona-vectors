# Activation I/O

Save and load activation vectors to/from disk for later analysis. This is the second
step in the persona vectors pipeline — after extracting activations with
`extract_activations()`, you save them here, then later load them for comparison and
analysis.

```
Dataset → Format Prompts → Extract Activations → Save → Analyze/Compare
```

---

## Why save activations?

Extracting activations is expensive — it requires running the model forward pass.
By saving activations to disk:

- **Reuse:** Run extraction once, analyze many times with different methods
- **Share:** Share activations with collaborators without them needing the model
- **Iterate:** Try different analysis techniques without re-extraction

---

## Directory Structure

> NOTE: This can be changed if we find something more resonable

```bash
artifacts/activations/
└── google/gemma-2-2b-it/                 # model_name
    └── templated/                        # prompt_variant
        └── persona_001/                  # persona_id
            ├── activations.safetensors   # (n_questions, n_layers, d_model)
            └── metadata.json             # {"questions": [...]}
```

| Component        | Description                                                 |
| ---------------- | ----------------------------------------------------------- |
| `root_dir`       | Parent directory (e.g., `./artifacts/activations`)          |
| `model_name`     | HuggingFace model identifier (e.g., `google/gemma-2-2b-it`) |
| `prompt_variant` | How prompts were formatted (e.g., `templated`, `raw`)       |
| `persona_id`     | Unique identifier for the persona                           |

---

## Save

After extracting activations, save them for later use:

```python
from src.activation_io import save_per_question_vectors

save_per_question_vectors(
    root_dir="./artifacts/activations",
    model_name="google/gemma-2-2b-it",
    prompt_variant="templated",
    persona_id=persona.id,
    per_question_vectors=activations,  # (n_questions, n_layers, d_model)
    questions=questions,                # list of question strings
)
```

The function automatically:

- Creates the directory structure if needed
- Saves activations as efficient `.safetensors` file
- Saves metadata (questions) in JSON for reference

---

## Load

Load saved activations for analysis:

```python
from src.activation_io import load_per_question_vectors

vectors, questions = load_per_question_vectors(
    root_dir="./artifacts/activations",
    model_name="google/gemma-2-2b-it",
    prompt_variant="templated",
    persona_id=persona.id,
)

# vectors: (n_questions, n_layers, d_model) tensor
# questions: list of question strings
```

---

## File Format

- **activations.safetensors**: Efficient binary format using PyTorch's safetensors,
  supports memory-mapped loading for large datasets
- **metadata.json**: Human-readable JSON with question strings for reference
