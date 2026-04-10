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
MODEL_NAME = "google/gemma-2-9b-it" if REMOTE else "google/gemma-2-2b-it"

print(f"Loading {MODEL_NAME}...")
if REMOTE:
    import nnsight

    # Meta device — no weights downloaded locally; execution happens on NDIF servers.
    # print(nnsight.ndif_status())
    # print(nnsight.ndif.compare())
    print(f"{MODEL_NAME} running: {nnsight.is_model_running(MODEL_NAME)}")
    # HACK: For now do it like this becuase of the bug
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
qa_pairs = dataset.get_qa(persona.id)  # full run
# qa_pairs = dataset.get_qa(persona.id)[:2]
print(f"Using {len(qa_pairs)} QA pairs for {persona.name}")
print(f"QIDs: {[qa.qid for qa in qa_pairs]}")

# %% Extract activations for all prompt variants
results = run_extraction(
    model=model,
    model_name=MODEL_NAME,
    persona=persona,
    qa_pairs=qa_pairs,
    variants=SUPPORTED_VARIANTS,
    mask_strategy=MaskStrategy.RESPONSE_FIRST,
    remote=REMOTE,
    verbose=True,
)

for r in results:
    print(f"Saved {r.variant} activations to {r.output_dir} ({r.n_questions} examples)")
