# %% Imports
import torch
from nnsight import LanguageModel
from tqdm.auto import tqdm

from src.activations import extract_activations
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


def generate_response(system_prompt: str, question: str) -> tuple[str, torch.Tensor]:
    """Generate a response and return the full formatted text plus a token mask."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    prompt, _ = format_messages(messages, tokenizer)

    # Tokenize prompt to know where generated tokens begin
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
    prompt_length = prompt_ids.shape[1]

    # NOTE: Generate response — hack for now; ideally responses come from real conversations
    with model.generate(prompt, max_new_tokens=N_TOKENS, do_sample=False) as tracer:
        result = tracer.result.save()

    # Decode only the newly generated tokens (exclude the input prompt)
    response_text = str(
        tokenizer.decode(result[0][prompt_length:], skip_special_tokens=True)
    )

    messages_full = messages + [{"role": "assistant", "content": response_text}]

    full_text, response_start_idx = format_messages(messages_full, tokenizer)

    token_mask = torch.arange(len(tokenizer(full_text).input_ids)) >= response_start_idx

    return full_text, token_mask


# %% Compare activations between templated_prompt and biography_md
def get_mean_activations(
    model, system_prompt: str, questions: list[str], label: str, verbose: bool = False
) -> list[torch.Tensor]:
    all_hs = []
    full_text: str = ""
    token_mask: torch.Tensor = torch.empty(0, dtype=torch.bool)

    # Generate response for each question with the given persona and extract activations
    for question in tqdm(questions, desc=label):
        full_text, token_mask = generate_response(system_prompt, question)
        all_hs.append(extract_activations(model, full_text, token_mask))

    # Show one example (the last one)
    if verbose:
        print(f"\n{full_text}\n")

    return [
        torch.stack([hs[i] for hs in all_hs]).mean(dim=0) for i in range(len(all_hs[0]))
    ]


# NOTE: Currently the different questions are flattened we can think of how to proceed with this and when to average
# But I would still keep things separate for each layer to have more flexibility for later
short_hidden_states = get_mean_activations(
    model, persona["templated_prompt"], EVAL_QUESTIONS, "short prompt"
)

long_hidden_states = get_mean_activations(
    model, persona["biography_md"], EVAL_QUESTIONS, "long prompt"
)

print(f"Hidden state shape per layer: {short_hidden_states[0].shape}")

# %% Compare activations across layers

# TODO: Work on this even empty prompt doesn't show much difference for know, we can do more testing here
# long_hidden_states = get_mean_activations(
#     model, "", EVAL_QUESTIONS, "long prompt", verbose=True
# )

# stack into torch tensors
short = torch.stack(short_hidden_states)  # (L, d_model)
long = torch.stack(long_hidden_states)  # (L, d_model)

# TODO: Fix the centering approach (just PoC and reminder for now)
# center per layer (along feature dimension)
# https://www.alignmentforum.org/posts/eLNo7b56kQQerCzp2/mech-interp-puzzle-1-suspiciously-similar-embeddings-in-gpt
# short = short - short.mean(dim=1, keepdim=True)
# long = long - long.mean(dim=1, keepdim=True)

# %% Plot cosine similarity across layers
persona_name = f"{persona['persona']['first_name']} {persona['persona']['last_name']}"
fig = plot_layer_similarity(
    short, long, title=f"Short vs Long Prompt — {persona_name}", show=True
)
