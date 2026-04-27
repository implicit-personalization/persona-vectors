"""Extract *per-token* activations for a persona — input to the SAE pipeline.

Saves the full residual stream at the requested layer(s) for each (persona,
QA) pair. Storage layout: ``artifacts/activations_pertoken/...``. See
``notebook_sae.py`` for the SAE feature-space analysis that consumes this.
"""

import torch
from dotenv import load_dotenv
from nnterp import StandardizedTransformer
from persona_data.environment import set_seed
from persona_data.synth_persona import SynthPersonaDataset
from rich.console import Console
from rich.table import Table

from persona_vectors.extraction_pertoken import run_pertoken_extraction

console = Console()

# %% Setup
load_dotenv()
torch.set_grad_enabled(False)
set_seed(1337)

# %% Configuration
# REMOTE = True
REMOTE = False
MODEL_NAME = "google/gemma-2-9b-it" if REMOTE else "google/gemma-2-2b-it"
LAYERS = [20]  # Single layer keeps disk usage to ~17 MB / persona / variant at 9B.

# %% Load model
print(f"Loading {MODEL_NAME}...")
model = StandardizedTransformer(MODEL_NAME)

# %% Load dataset, pick a persona, take a few QA pairs
dataset = SynthPersonaDataset(sample_size=1)
persona = dataset[0]
qa_pairs = dataset.get_qa(persona.id)[:8]
# qa_pairs = dataset.get_qa(persona.id)

table = Table(title="Per-token extraction config")
table.add_column("Property", style="cyan")
table.add_column("Value", style="magenta")
table.add_row("Model", MODEL_NAME)
table.add_row("Persona", persona.name)
table.add_row("QA pairs", str(len(qa_pairs)))
table.add_row("Layers", str(LAYERS))
console.print(table)

# %% Run per-token extraction for both prompt variants
results = run_pertoken_extraction(
    model=model,
    model_name=MODEL_NAME,
    persona=persona,
    qa_pairs=qa_pairs,
    variants=("templated", "biography"),
    layers=LAYERS,
    remote=REMOTE,
)

for r in results:
    print(
        f"Saved {r.variant} ({r.n_questions} questions × {len(r.layers)} layers) → {r.output_dir}"
    )
