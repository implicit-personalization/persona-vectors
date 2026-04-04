import torch
from dotenv import load_dotenv
from persona_data.environment import get_artifacts_dir
from persona_data.synth_persona import SynthPersonaDataset
from rich.console import Console
from rich.table import Table

from persona_vectors.activation_io import load_per_question_vectors
from persona_vectors.steering import compute_steering_vector, save_steering_vector

console = Console()

# %% Setup code
load_dotenv()
torch.set_grad_enabled(False)

# %% Configuration
# Use 9b for remote (production), 2b for local testing
# REMOTE = False
REMOTE = True
MODEL_NAME = "google/gemma-2-9b-it" if REMOTE else "google/gemma-2-2b-it"
STEER_LAYER = 20

# %% Load dataset
dataset = SynthPersonaDataset()
first_persona = dataset[0]

dataset_table = Table(title="Dataset")
dataset_table.add_column("Property", style="cyan")
dataset_table.add_column("Value", style="magenta")
dataset_table.add_row("Total Personas", str(len(dataset)))
dataset_table.add_row("First Persona", first_persona.name)
dataset_table.add_row("Age", str(first_persona.persona["age"]))
console.print(dataset_table)

persona = first_persona
ACTIVATIONS_DIR = get_artifacts_dir() / "activations"

# %% Load activations for both variants
results = {}
for variant in ["templated", "biography"]:
    per_question_activations, _ = load_per_question_vectors(
        root_dir=ACTIVATIONS_DIR,
        model_name=MODEL_NAME,
        prompt_variant=variant,
        persona_id=persona.id,
    )

    results[variant] = per_question_activations

print(f"Biography activations shape: {results['biography'].shape}")
print(f"Templated activations shape: {results['templated'].shape}")

# %% Compute steering vector
sv_dict = compute_steering_vector(
    persona_id=persona.id,
    model_name=MODEL_NAME,
    layer_idx=STEER_LAYER,
    activations_dir=ACTIVATIONS_DIR,
)

# %% Inspect steering vector
if sv_dict:
    sv = sv_dict["steering_vector"]
    print(f"\nSteering vector shape: {sv.shape}")
    print(f"Suggested alpha: {sv_dict['suggested_alpha']:.4f}")
    print(f"L2 norm: {sv.squeeze().norm().item():.6f}")
    print(f"QA pairs used: {sv_dict['n_qa_pairs']}")

# %% Save steering vector as safetensors
if sv_dict:
    out_dir = get_artifacts_dir() / "vectors" / persona.id
    save_steering_vector(sv_dict, out_dir)
