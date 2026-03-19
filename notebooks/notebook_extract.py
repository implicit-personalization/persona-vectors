# %% Imports
import nnsight
import torch
from nnsight import LanguageModel
from rich.console import Console
from rich.table import Table
from tqdm.auto import tqdm

from src.activation_io import save_per_question_vectors
from src.activations import extract_activations
from src.environment import get_artifacts_dir, load_env, set_seed
from src.prompt_format import format_messages
from src.synth_persona_io import QAPair, SynthPersonaDataset

console = Console()

# %% Setup code
load_env()  # Load .env (NDIF_API_KEY, HF_HOME, etc.) before anything else
torch.set_grad_enabled(False)
set_seed(1337)

# %% Setting up the model
# Use 9b for remote (production), 2b for local testing
# REMOTE = False
REMOTE = True
MODEL_NAME = "google/gemma-2-9b-it" if REMOTE else "google/gemma-2-2b-it"

print(f"Loading {MODEL_NAME}...")
if REMOTE:
    # Meta device — no weights downloaded locally; execution happens on NDIF servers.
    # print(nnsight.ndif_status())
    # print(nnsight.ndif.compare())
    print(f"{MODEL_NAME} running: {nnsight.is_model_running(MODEL_NAME)}")
    model = LanguageModel(MODEL_NAME)
else:
    model = LanguageModel(MODEL_NAME, dtype="auto", device_map="auto", dispatch=True)

tokenizer = model.tokenizer

NUM_LAYERS = model.config.num_hidden_layers
D_MODEL = model.config.hidden_size

model_table = Table(title="Model Config")
model_table.add_column("Property", style="cyan")
model_table.add_column("Value", style="magenta")
model_table.add_row("Model", MODEL_NAME)
model_table.add_row("Layers", str(NUM_LAYERS))
model_table.add_row("Hidden Size", str(D_MODEL))
console.print(model_table)

# %% Load dataset from HuggingFace
dataset = SynthPersonaDataset()
first_persona = dataset[0]

dataset_table = Table(title="Dataset")
dataset_table.add_column("Property", style="cyan")
dataset_table.add_column("Value", style="magenta")
dataset_table.add_row("Total Personas", str(len(dataset)))
dataset_table.add_row("First Persona", first_persona.name)
dataset_table.add_row("Age", str(first_persona.persona["age"]))
console.print(dataset_table)

# %% Pick persona and get QA pairs
persona = first_persona
# qa_pairs = dataset.get_qa(persona.id)[:2] # for short example
qa_pairs = dataset.get_qa(persona.id)
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
    """Extract masked mean activations and store lightweight metadata."""
    full_texts: list[str] = []
    token_masks: list[torch.Tensor] = []
    all_questions: list[str] = []

    for qa in tqdm(qa_pairs, desc=label):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": qa.question},
            {"role": "assistant", "content": qa.answer},
        ]
        full_prompt, answer_start = format_messages(messages, tokenizer)
        seq_len = tokenizer(full_prompt, return_tensors="pt").input_ids.shape[1]

        full_texts.append(full_prompt)
        token_masks.append(torch.arange(seq_len) >= answer_start)
        all_questions.append(qa.question)

    all_hs = extract_activations(model, full_texts, token_masks, remote=remote)

    artifact_dir = save_per_question_vectors(
        root_dir=ACTIVATIONS_DIR,
        model_name=model_name,
        prompt_variant=prompt_variant,
        persona_id=persona_id,
        per_question_vectors=all_hs,
        questions=all_questions,
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
