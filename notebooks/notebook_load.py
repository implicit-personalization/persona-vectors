# %% Imports
import torch
from persona_data.environment import get_artifacts_dir, load_env, set_seed
from persona_data.synth_persona import SynthPersonaDataset
from rich.console import Console
from rich.table import Table

from src.activation_io import load_per_question_vectors
from src.plots import plot_layer_similarity

console = Console()

# %% Setup code
load_env()
torch.set_grad_enabled(False)
set_seed(1337)

# %% Configuration
# Use 9b for remote (production), 2b for local testing
# REMOTE = False
REMOTE = True
MODEL_NAME = "google/gemma-2-9b-it" if REMOTE else "google/gemma-2-2b-it"

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

# %% Load activations and use stored metadata
results = {}
for variant in ["templated", "biography"]:
    per_question_activations, _ = load_per_question_vectors(
        root_dir=ACTIVATIONS_DIR,
        model_name=MODEL_NAME,
        prompt_variant=variant,
        persona_id=persona.id,
    )

    results[variant] = per_question_activations.mean(dim=0)

short_hidden_states = results["templated"]
long_hidden_states = results["biography"]

print(f"Hidden state shape: {short_hidden_states.shape}")

# %% Plot cosine similarity across layers
# TODO: Work on this even empty prompt doesn't show much difference for know, we can do more testing here
# long_hidden_states = get_mean_activations(
#     model, "", EVAL_QUESTIONS, "long prompt", verbose=True
# )

# TODO: Fix the centering approach (just PoC and reminder for now)
# center per layer (along feature dimension)
# https://www.alignmentforum.org/posts/eLNo7b56kQQerCzp2/mech-interp-puzzle-1-suspiciously-similar-embeddings-in-gpt
# short = short - short.mean(dim=1, keepdim=True)
# long = long - long.mean(dim=1, keepdim=True)

fig = plot_layer_similarity(
    short_hidden_states,
    long_hidden_states,
    title=f"Templated vs Biography — {persona.name}",
    show=True,
)
