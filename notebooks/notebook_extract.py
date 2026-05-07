import torch
from dotenv import load_dotenv
from nnterp import StandardizedTransformer
from persona_data.environment import set_seed
from persona_data.synth_persona import SynthPersonaDataset
from rich.console import Console
from rich.table import Table

from persona_vectors.artifacts import PERSONA_VARIANTS
from persona_vectors.extraction import MaskStrategy, run_extraction

console = Console()

# %% Setup code
# Load .env (NDIF_API_KEY, HF_HOME, etc.) before anything else
load_dotenv()
torch.set_grad_enabled(False)
set_seed(1337)

# %% Setting up the model
# Use 9b for remote (production), 2b for local testing
# REMOTE = False
REMOTE = True
MODEL_NAME = "google/gemma-2-9b-it" if REMOTE else "google/gemma-2-2b-it"

print(f"Loading {MODEL_NAME}...")
if REMOTE:
    import nnsight

    # Meta device — no weights downloaded locally; execution happens on NDIF servers.
    # print(nnsight.ndif_status())
    # print(nnsight.ndif.compare())
    print(f"{MODEL_NAME} running: {nnsight.is_model_running(MODEL_NAME)}")
    # Work around the current remote initialization issue.
    # model = StandardizedTransformer(MODEL_NAME, remote=True)
    model = StandardizedTransformer(MODEL_NAME)
else:
    model = StandardizedTransformer(MODEL_NAME)

tokenizer = model.tokenizer

# NOTE: This is much cleaner with nnterp
NUM_LAYERS = model.num_layers
D_MODEL = model.hidden_size

model_table = Table(title="Model Config")
model_table.add_column("Property", style="cyan")
model_table.add_column("Value", style="magenta")
model_table.add_row("Model", MODEL_NAME)
model_table.add_row("Layers", str(NUM_LAYERS))
model_table.add_row("Hidden Size", str(D_MODEL))
console.print(model_table)

# %% Load dataset and select runs (small persona-only smoke run)
# SynthPersonaDataset(sample_size=N) keeps the leading N personas from the
# HF JSONL. We then take the train side of train_test_split per persona:
#   - train: individual FRQs (free-response), leakage-filtered against the MCQ bank
#   - test:  shared MCQs (same item bank for every persona) -- not used here
# n_train caps the train slice; pass None for every non-leaking FRQ.
N_TRAIN = 8
dataset = SynthPersonaDataset(sample_size=3)
runs = [
    (persona, dataset.train_test_split(persona.id, n_train=N_TRAIN)[0])
    for persona in dataset
]
runs = [(p, qa) for p, qa in runs if qa]  # drop personas with no train QAs

# To extract the Assistant baseline or a custom explicit list in one run:
# from persona_vectors.extraction import select_personas_with_qa
# runs = select_personas_with_qa(
#     SynthPersonaDataset(),
#     persona_ids=["baseline_assistant", "<UUID>"],
# )

dataset_table = Table(title="Dataset")
dataset_table.add_column("Property", style="cyan")
dataset_table.add_column("Value", style="magenta")
dataset_table.add_row("Total Personas", str(len(dataset)))
dataset_table.add_row("Train cap (n_train)", str(N_TRAIN))
for persona, qa_pairs in runs:
    dataset_table.add_row(persona.name, f"{len(qa_pairs)} train QA")
console.print(dataset_table)

# %% Extract activations for all prompt variants
# For each persona, run a forward pass per QA pair across both prompt variants
# (templated, biography), then save the response-token mean -- averaged over
# masked tokens *and* over QA pairs -- as a single (num_layers, hidden_size)
# tensor per (variant, persona). Steering uses the diff at STEER_LAYER.
# RUN_NAME = "run_01"
MASK_STRATEGY = MaskStrategy.ANSWER_MEAN
for persona, qa_pairs in runs:
    print(f"\n→ {persona.name}: {[qa.qid for qa in qa_pairs]}")
    for r in run_extraction(
        model=model,
        model_name=MODEL_NAME,
        qa_pairs=qa_pairs,
        variants=PERSONA_VARIANTS,
        persona=persona,
        mask_strategy=MASK_STRATEGY,
        remote=REMOTE,
        verbose=True,
        # activations_dir=f"artifacts/activations/{RUN_NAME}",
    ):
        print(f"Saved {r.variant} → {r.output_dir} ({r.n_questions} examples)")
