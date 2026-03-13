# %% Imports
import torch

from src.activation_io import load_per_question_activations
from src.environment import get_artifacts_dir, load_env, set_seed
from src.plots import plot_layer_similarity
from src.synth_persona_io import SynthPersonaDataset

# %% Setup code
load_env()
torch.set_grad_enabled(False)
set_seed(1337)

# %% Configuration
# Use 9b for remote (production), 2b for local testing
REMOTE = True
MODEL_NAME = "google/gemma-2-9b-it" if REMOTE else "google/gemma-2-2b-it"

# %% Load dataset
dataset = SynthPersonaDataset()
print(f"Loaded {len(dataset)} personas")
first_persona = dataset[0]
print(f"Persona 0: {first_persona.name} Age: {first_persona.persona['age']}")

persona = first_persona
ACTIVATIONS_DIR = get_artifacts_dir() / "activations"

# %% Load activations and use stored metadata
results = {}
for variant in ["templated", "biography"]:
    per_question_activations, _ = load_per_question_activations(
        root_dir=ACTIVATIONS_DIR,
        model_name=MODEL_NAME,
        prompt_variant=variant,
        persona_id=persona.id,
    )

    # Activations are already reduced to the chosen token span during extraction,
    # so loading is now just averaging over questions.
    results[variant] = torch.stack(per_question_activations, dim=0).mean(dim=0)

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
