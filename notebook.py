# %% Imports
import torch
from nnsight import LanguageModel

from src.activations import get_mean_activations
from src.environment import set_seed
from src.format import format_messages
from src.load import load_personas
from src.plots import plot_layer_similarity

# %% Setup code
torch.set_grad_enabled(False)
set_seed(1337)

# %% Setting up the model
# NOTE: The model is a small model which is easy to run and test locally.
# I would keep it like that for this notebook. And keeping inference local for simplicity
# Later I would focus on creating a notebook which instead loads directly from saved activations
# And instead focuses on plotting and other analyses
MODEL_NAME = "google/gemma-2-2b-it"
DTYPE = torch.bfloat16

print(f"Loading {MODEL_NAME}...")

model = LanguageModel(
    MODEL_NAME,
    dtype=DTYPE,
    device_map="auto",
    attn_implementation="eager",  # This was present in Arena I didn't change it for now
)
tokenizer = model.tokenizer

NUM_LAYERS = model.config.num_hidden_layers
D_MODEL = model.config.hidden_size
print(f"Model loaded with {NUM_LAYERS} layers")
print(f"Hidden size: {D_MODEL}")

# %% Definition of some basic questions

# HACK: For now we take some simple questions to check things are working
EVAL_QUESTIONS = [
    "What advice would you give to someone starting a new chapter in their life?",
    "How do you view the relationship between knowledge and wisdom?",
    "What do you think about the nature of truth?",
    "How should someone approach making difficult decisions?",
    "What role does creativity play in problem-solving?",
    "How do you see the balance between tradition and progress?",
    "What matters most when building trust with others?",
    "How do you think about the passage of time?",
    "What would you say to someone feeling lost or uncertain?",
    "How do you approach understanding something complex?",
    "What do you think about the nature of change?",
    "How should one deal with failure or setbacks?",
    "What role does intuition play in understanding?",
    "How do you view the relationship between the individual and society?",
    "What do you think makes something meaningful?",
]

# NOTE: Work with a subset for faster inference
EVAL_QUESTIONS = EVAL_QUESTIONS[:2]

print(f"Defined {len(EVAL_QUESTIONS)} evaluation questions")

# %% Getting the personas (Example )
personas = load_personas()
print(f"Loaded {len(personas)} personas")

first_persona = personas[0]["persona"]
print(
    f"Persona 0: {first_persona['first_name']} {first_persona['last_name']} Age: {first_persona['age']}"
)

# %% Test: Generate responses for the different contexts
persona = personas[0]
N_TOKENS = 50


# HACK: Generating the response here is a hack for now — later it would be
# nice to use responses from actual conversations or a dedicated pipeline.


def generate_response(system_prompt: str, question: str) -> tuple[str, int]:
    """Generate a response and return the full formatted text plus the response start index."""

    # NOTE: For now just replicating the approach from the original implementation mostly.
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    prompt, _ = format_messages(messages, tokenizer)

    # NOTE: Generate response — hack for now; ideally responses come from real conversations
    with model.generate(prompt, max_new_tokens=N_TOKENS, do_sample=False) as tracer:
        result = tracer.result.save()

    response_text = tokenizer.decode(result[0], skip_special_tokens=True)

    # NOTE: For now just copying what they have done
    messages_full = messages + [{"role": "assistant", "content": response_text}]
    full_text, response_start_idx = format_messages(messages_full, tokenizer)

    return full_text, response_start_idx


# %% Compare activations between templated_prompt and biography_md

# NOTE: Currently the different questions are flattened we can think of how to proceed with this and when to average
# But I would still keep things separate for each layer to have more flexibility for later
short_hidden_states = get_mean_activations(
    model,
    persona["templated_prompt"],
    EVAL_QUESTIONS,
    "short prompt",
    generate_response,
)

long_hidden_states = get_mean_activations(
    model,
    persona["biography_md"],
    EVAL_QUESTIONS,
    "long prompt",
    generate_response,
)

print(f"Hidden state shape per layer: {short_hidden_states[0].shape}")

# %% Compare activations across layers

# stack into torch tensors
short = torch.stack(short_hidden_states)  # (L, d_model)
long = torch.stack(long_hidden_states)  # (L, d_model)

# center per layer (along feature dimension)
# https://www.alignmentforum.org/posts/eLNo7b56kQQerCzp2/mech-interp-puzzle-1-suspiciously-similar-embeddings-in-gpt
short = short - short.mean(dim=1, keepdim=True)
long = long - long.mean(dim=1, keepdim=True)

# %% Plot cosine similarity across layers
persona_name = f"{persona['persona']['first_name']} {persona['persona']['last_name']}"
fig = plot_layer_similarity(
    short, long, title=f"Short vs Long Prompt — {persona_name}", show=True
)
