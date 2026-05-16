import torch
from dotenv import load_dotenv
from nnterp import StandardizedTransformer
from persona_data.environment import set_seed
from persona_data.synth_persona import SynthPersonaDataset
from rich.console import Console
from rich.table import Table

from persona_vectors.artifacts import SUPPORTED_VARIANTS
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
MODEL_NAME = "meta-llama/Llama-3.1-70B-Instruct" if REMOTE else "google/gemma-2-2b-it"

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

# %% Load dataset and select runs
N_TRAIN = 8
dataset = SynthPersonaDataset(sample_size=3)
runs = [
    (persona, dataset.train_test_split(persona.id, n_train=N_TRAIN)[0])
    for persona in dataset
]
runs = [(p, qa) for p, qa in runs if qa]  # drop personas with no train QAs

# The Assistant baseline is just another row in the JSONL load it by
# dropping sample_size, or pull it explicitly with dataset.get_persona("baseline_assistant").
dataset_table = Table(title="Dataset")
dataset_table.add_column("Property", style="cyan")
dataset_table.add_column("Value", style="magenta")
dataset_table.add_row("Personas loaded", str(len(dataset)))
dataset_table.add_row("Train cap (n_train)", str(N_TRAIN))
for persona, qa_pairs in runs:
    dataset_table.add_row(persona.name, f"{len(qa_pairs)} train QA")
console.print(dataset_table)

# %% Extract activations for all prompt variants
# For each persona, run a forward pass per QA pair across both prompt variants
# (templated, biography), then save the response-token mean -- averaged over
# masked tokens over QA pairs -- as a single (num_layers, hidden_size) tensor per (variant, persona).
RUN_NAME = "run_01"
MASK_STRATEGY = MaskStrategy.ANSWER_MEAN
for persona, qa_pairs in runs:
    print(f"\n→ {persona.name}: {[qa.qid for qa in qa_pairs]}")
    for r in run_extraction(
        model=model,
        model_name=MODEL_NAME,
        qa_pairs=qa_pairs,
        variants=SUPPORTED_VARIANTS,
        persona=persona,
        mask_strategy=MASK_STRATEGY,
        remote=REMOTE,
        verbose=True,
        activations_dir=f"artifacts/activations/{RUN_NAME}",
    ):
        print(f"Saved {r.variant} → {r.output_dir} ({r.n_questions} examples)")
