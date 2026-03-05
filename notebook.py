# %% Imports
import nnsight
import torch
from nnsight import LanguageModel
from tqdm.auto import tqdm

from src.activations import extract_activations
from src.environment import load_env, set_seed
from src.format import format_messages
from src.load import load_personas
from src.plots import plot_layer_similarity

# %% Setup code

# Load .env (NDIF_API_KEY, HF_HOME, etc.) before anything else
load_env()
torch.set_grad_enabled(False)
set_seed(1337)

# %% Setting up the model
# Set REMOTE=True to run inference on NDIF servers instead of locally.
# Requires NDIF_API_KEY in .env. The model loads on the meta device (no local GPU needed).
REMOTE = False
# REMOTE = True

# MODEL_NAME = "google/gemma-2-9b-it"
MODEL_NAME = "google/gemma-2-9b-it"
DTYPE = torch.bfloat16

print(f"Loading {MODEL_NAME}...")
if REMOTE:
    # Meta device — no weights downloaded locally; execution happens on NDIF servers.
    # print(nnsight.ndif_status())
    # print(nnsight.ndif.compare())
    print(f"{MODEL_NAME} running: {nnsight.is_model_running(MODEL_NAME)}")
    model = LanguageModel(MODEL_NAME)
else:
    # NOTE: The model is a small model which is easy to run and test locally.
    # I would keep it like that for this notebook. And keeping inference local for simplicity
    MODEL_NAME = "google/gemma-2-2b-it"
    model = LanguageModel(
        MODEL_NAME,
        dtype=DTYPE,
        device_map="auto",
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
# FIX: With more then 1 qeustion since I'm batching it crashes on the Remote (OOM)
# I think it leverages also KV caching so it makes sense with the biography prompt I guess
EVAL_QUESTIONS = EVAL_QUESTIONS[:1]

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
    with model.generate(
        prompt, max_new_tokens=N_TOKENS, do_sample=False, remote=REMOTE
    ) as tracer:
        result = tracer.result.save()

    # Decode only the newly generated tokens (exclude the input prompt)
    response_text = str(
        tokenizer.decode(result[0][prompt_length:], skip_special_tokens=True)
    )

    messages_full = messages + [{"role": "assistant", "content": response_text}]
    full_text, response_start_idx = format_messages(messages_full, tokenizer)

    # NOTE: Mask tokens in the responses (all of them) -> This can be flexibly changed to save just last the last token
    token_mask = torch.arange(len(tokenizer(full_text).input_ids)) >= response_start_idx

    return full_text, token_mask


# %% Compare activations between templated_prompt and biography_md
def get_mean_activations(
    model,
    system_prompt: str,
    questions: list[str],
    label: str,
    verbose: bool = False,
    remote: bool = False,
) -> torch.Tensor:
    full_texts: list[str] = []
    token_masks: list[torch.Tensor] = []

    # Generate response for each question with the given persona
    for question in tqdm(questions, desc=label):
        full_text, token_mask = generate_response(system_prompt, question)
        full_texts.append(full_text)
        token_masks.append(token_mask)

    # Show one example (the last one)
    if verbose:
        print(f"\n{full_texts[-1] = }\n")

    # Single batched forward pass over all questions.
    # token_masks are built against unpadded sequences; extract_activations
    # tokenizes with padding=True and right-aligns them to handle nnsight's left-padding.
    all_hs = extract_activations(model, full_texts, token_masks, remote=remote)

    # (n_questions, L, d_model) -> (L, d_model)
    return all_hs.mean(dim=0)


# NOTE: Currently the different questions are flattened we can think of how to proceed with this and when to average
# But I would still keep things separate for each layer to have more flexibility for later
short_hidden_states = get_mean_activations(
    model, persona["templated_prompt"], EVAL_QUESTIONS, "short prompt", remote=REMOTE
)

long_hidden_states = get_mean_activations(
    model, persona["biography_md"], EVAL_QUESTIONS, "long prompt", remote=REMOTE
)

print(f"Hidden state shape: {short_hidden_states.shape}")

# %% Compare activations across layers

# TODO: Work on this even empty prompt doesn't show much difference for know, we can do more testing here
# long_hidden_states = get_mean_activations(
#     model, "", EVAL_QUESTIONS, "long prompt", verbose=True
# )

# TODO: Fix the centering approach (just PoC and reminder for now)
# center per layer (along feature dimension)
# https://www.alignmentforum.org/posts/eLNo7b56kQQerCzp2/mech-interp-puzzle-1-suspiciously-similar-embeddings-in-gpt
# short = short - short.mean(dim=1, keepdim=True)
# long = long - long.mean(dim=1, keepdim=True)

# %% Plot cosine similarity across layers
persona_name = f"{persona['persona']['first_name']} {persona['persona']['last_name']}"
fig = plot_layer_similarity(
    short_hidden_states,
    long_hidden_states,
    title=f"Short vs Long Prompt — {persona_name}",
    show=True,
)
