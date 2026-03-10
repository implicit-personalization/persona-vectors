# %% Imports
import nnsight
import torch
from nnsight import LanguageModel
from tqdm.auto import tqdm

from src.activation_io import save_per_question_activations
from src.activations import extract_activations
from src.environment import get_artifacts_dir, load_env, set_seed
from src.prompt_format import format_messages
from src.synth_persona_io import QAPair, SynthPersonaDataset

# %% Setup code
load_env()  # Load .env (NDIF_API_KEY, HF_HOME, etc.) before anything else
torch.set_grad_enabled(False)
set_seed(1337)

# %% Setting up the model
# Use 9b for remote (production), 2b for local testing
# REMOTE = False
REMOTE = True

# NOTE: This is for ease of use and testing on my computer and remotly at the same time
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
dataset = SynthPersonaDataset()
print(f"Loaded {len(dataset)} personas")
first_persona = dataset[0]
print(f"Persona 0: {first_persona.name} Age: {first_persona.persona['age']}")

# %% Pick persona and get QA pairs
persona = first_persona

# NOTE: Work with a subset for faster inference
qa_pairs = dataset.get_qa(persona.id)[:2]
print(f"Using {len(qa_pairs)} QA pairs for {persona.name}")
print(f"QIDs: {[qa.qid for qa in qa_pairs]}")

# %% Options for activation extraction
ACTIVATIONS_DIR = get_artifacts_dir() / "activations"


def extract_variant_activations(
    model,
    model_name: str,
    persona_id: str,
    prompt_variant: str,
    system_prompt: str,
    qa_pairs: list[QAPair],
    label: str,
    remote: bool = False,
):
    """Extract activations and store rich metadata with token indices."""
    full_texts: list[str] = []
    all_metadata: list[dict] = []

    for qa in tqdm(qa_pairs, desc=label):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": qa.question},
            {"role": "assistant", "content": qa.answer},
        ]
        full_prompt, answer_start = format_messages(messages, tokenizer)
        seq_len = tokenizer(full_prompt, return_tensors="pt").input_ids.shape[1]

        full_texts.append(full_prompt)

        # HACK: For now we only store the assistant answer span because that is
        # the only boundary used in the current analysis notebook.
        all_metadata.append(
            {
                "qid": qa.qid,
                "question": qa.question,
                "answer": qa.answer,
                "seq_len": seq_len,
                "answer_start": answer_start,
                "answer_end": seq_len,
            }
        )

    all_hs = extract_activations(model, full_texts, remote=remote)

    artifact_dir = save_per_question_activations(
        root_dir=ACTIVATIONS_DIR,
        model_name=model_name,
        prompt_variant=prompt_variant,
        persona_id=persona_id,
        per_question_activations=all_hs,
        metadata=all_metadata,
    )
    print(
        f"Saved {prompt_variant} activations to {artifact_dir} ({len(qa_pairs)} examples)"
    )


# %% Extract activations for different prompt variants
variants = [
    ("templated", persona.templated_prompt, "templated prompt"),
    ("biography", persona.biography_md, "biography prompt"),
]

for variant_name, system_prompt, label in variants:
    extract_variant_activations(
        model,
        MODEL_NAME,
        persona.id,
        variant_name,
        system_prompt,
        qa_pairs,
        label,
        remote=REMOTE,
    )

print("Extraction complete!")
