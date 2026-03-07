# %% Imports
import nnsight
import torch
from nnsight import LanguageModel
from tqdm.auto import tqdm

from src.activation_io import load_per_question_vectors, save_per_question_vectors
from src.activations import extract_activations
from src.environment import get_artifacts_dir, load_env, set_seed
from src.persona_io import load_personas, load_qa_pairs
from src.plots import plot_layer_similarity
from src.prompt_format import format_messages

# %% Setup code

# Load .env (NDIF_API_KEY, HF_HOME, etc.) before anything else
load_env()
torch.set_grad_enabled(False)
set_seed(1337)

# %% Setting up the model
# Set REMOTE=True to run inference on NDIF servers instead of locally.
# Requires NDIF_API_KEY in .env. The model loads on the meta device (no local GPU needed).
# REMOTE = False
REMOTE = True

MODEL_NAME = "google/gemma-2-9b-it" if REMOTE else "google/gemma-2-2b-it"
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

# %% Load dataset from HuggingFace
HF_REPO = "implicit-personalization/synth-persona"
personas = load_personas(from_hf=True, hf_repo=HF_REPO)
qa_by_persona = load_qa_pairs(from_hf=True, hf_repo=HF_REPO)  # dict[id -> list[QAPair]]
print(f"Loaded {len(personas)} personas")

first_persona = personas[0]
print(f"Persona 0: {first_persona.name} Age: {first_persona.persona['age']}")

# %% Pick persona and build evaluation questions from its QA pairs
# NOTE: Work with a subset for faster inference
EVAL_QUESTIONS = [qa.question for qa in qa_by_persona[first_persona.id]][:2]
print(f"Using {len(EVAL_QUESTIONS)} evaluation questions for {first_persona.name}")

# %% Test: Generate responses for the different contexts
persona = first_persona
N_TOKENS = 50
ACTIVATIONS_DIR = get_artifacts_dir() / "activations"


# HACK: Generating the response here is a hack for now — later it would be
# nice to use responses from actual conversations or a dedicated pipeline.
def generate_response(
    system_prompt: str,
    question: str,
    remote: bool = False,
) -> tuple[str, torch.Tensor]:
    """Generate a response and return the full formatted text plus a token mask."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    prompt, _ = format_messages(messages, tokenizer)

    # Tokenize prompt to know where generated tokens begin
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
    prompt_length = prompt_ids.shape[1]

    # FIX: Instead of this it should use the actual answers for the questsions
    # NOTE: Generate response — hack for now; ideally responses come from real conversations
    with model.generate(
        prompt, max_new_tokens=N_TOKENS, do_sample=False, remote=remote
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
    model_name: str,
    persona_id: str,
    prompt_variant: str,
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
        full_text, token_mask = generate_response(
            system_prompt,
            question,
            remote=remote,
        )
        full_texts.append(full_text)
        token_masks.append(token_mask)

    # Show one example (the last one)
    if verbose:
        print(f"\n{full_texts[-1]}\n")

    all_hs = extract_activations(
        model,
        full_texts,
        token_masks,
        remote=remote,
    )

    save_per_question_vectors(
        root_dir=ACTIVATIONS_DIR,
        model_name=model_name,
        prompt_variant=prompt_variant,
        persona_id=persona_id,
        per_question_vectors=all_hs,
        questions=questions,
    )

    loaded_hs, loaded_questions = load_per_question_vectors(
        root_dir=ACTIVATIONS_DIR,
        model_name=model_name,
        prompt_variant=prompt_variant,
        persona_id=persona_id,
    )

    # NOTE: Some sanity checks
    if loaded_questions != questions:
        raise ValueError("loaded questions do not match saved questions")
    if not torch.equal(loaded_hs, all_hs):
        raise ValueError("loaded activations do not match saved activations")

    # (n_questions, L, d_model) -> (L, d_model)
    return loaded_hs.mean(dim=0)


# NOTE: Currently the different questions are flattened we can think of how to proceed with this and when to average
# But I would still keep things separate for each layer to have more flexibility for later
short_hidden_states = get_mean_activations(
    model,
    MODEL_NAME,
    persona.id,
    "templated",
    persona.templated_prompt,
    EVAL_QUESTIONS,
    "short prompt",
    remote=REMOTE,
)

long_hidden_states = get_mean_activations(
    model,
    MODEL_NAME,
    persona.id,
    "biography",
    persona.biography_md,
    EVAL_QUESTIONS,
    "long prompt",
    remote=REMOTE,
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
fig = plot_layer_similarity(
    short_hidden_states,
    long_hidden_states,
    title=f"Short vs Long Prompt — {persona.name}",
    show=True,
)
