# Generating Persona Steering Vectors

## Quick Start

### 1. Configure HuggingFace Authentication

The SynthPersona dataset is private. You need to authenticate:

```bash
huggingface-cli login
```

Then paste your HuggingFace API token (get it from https://huggingface.co/settings/tokens)

Alternatively, set the environment variable:
```bash
export HF_TOKEN=<your-token>  # On Linux/Mac
set HF_TOKEN=<your-token>     # On Windows PowerShell
```

### 2. Test Dataset Loading

Verify you can access the dataset:

```bash
cd persona-vectors
python scripts/test_dataset.py
```

You should see:
```
✓ Successfully loaded dataset: SynthPersonaDataset(n_personas=5)
  Personas: 5
  First persona: ...
```

### 3. List Available Personas

```bash
python scripts/run_generate_vectors.py --list
```

Output:
```
Available personas (5 total):

  [0] First Name Last Name
      ID: 0023952f-142e-434b-82e2-7a7451b7c55f
      QA pairs: 52
```

### 4. Generate Vectors

**For one persona:**
```bash
python scripts/run_generate_vectors.py 0023952f-142e-434b-82e2-7a7451b7c55f
```

**For all personas:**
```bash
python scripts/run_generate_vectors.py --all
```

Vectors are saved to: `artifacts/vectors/{persona_id}.pt`

Each `.pt` file contains:
- `steering_vector`: shape [1, 1, 3584] (for Gemma-2-9b-it at layer 20)
- `suggested_alpha`: scaling coefficient for steering
- `persona_id`, `layer`, `model_id`: metadata

## What's Happening

For each persona and QA pair:

1. **Positive trace**: Run model with persona biography as system prompt + QA pair
2. **Negative trace**: Run model with generic prompt + same QA pair
3. **Extract activations**: Get hidden states at specified layer for response tokens
4. **Compute vector**: `steering_vector = mean(pos_activations) - mean(neg_activations)`
5. **Estimate alpha**: `20 × mean_rms(neg_activations) / ||steering_vector||`

The steering vector encodes how adding it to the residual stream at inference makes the model's behavior shift toward the persona.

## Dataset Schema

**dataset_personas.jsonl** (one per line):
```json
{
  "id": "persona-uuid",
  "persona": { "first_name": "...", "last_name": "...", ... },
  "templated_prompt": "Age: 34\nOccupation: Nurse\n...",
  "biography_md": "Long-form persona narrative..."
}
```

**dataset_qa.jsonl** (one per line):
```json
{
  "id": "persona-uuid",
  "qid": "uuid-q001",
  "type": "explicit|implicit",
  "question": "Where did you grow up?",
  "answer": "Little Rock, Arkansas.",
  "difficulty": 1,
  ...
}
```

## Troubleshooting

### "Failed to download SynthPersona dataset"
- Make sure `huggingface-cli login` was run
- Verify your token has access to the implicit-personalization organization
- Try: `huggingface-cli whoami` to see your authenticated user

### Remote tracing fails
- Ensure `NDIF_API_KEY` is set (for nnsight remote compute)
- Check that you have internet connectivity

### Out of memory
- Reduce batch size (not yet configurable, but can be added)
- Use a smaller model
- Process one persona at a time
