#!/usr/bin/env python

# # Deconfounded trait vectors from minimal-pair attribute swaps
#
# Mirrors `notebook_extract.py`, but instead of one vector per persona we build
# one vector per **binary attribute**: swap only that attribute on each persona
# (re-rendering the whole templated view, via `persona_data.templated`), extract
# both views, and average the within-pair activation delta. Everything that did
# not change cancels, so the direction isolates the attribute instead of
# absorbing whatever co-occurs with it across the population.
#
# Closing cells compare the **trait-cosine** matrix against the **co-occurrence**
# (Cramér's V) matrix already built in `notebooks/unsupervised`: high
# co-occurrence with low trait-cosine = the minimal-pair extraction successfully
# deconfounded that pair.

# %% Setup
import numpy as np
import plotly.graph_objects as go
import torch
from dotenv import load_dotenv
from nnterp import StandardizedTransformer
from persona_data.environment import set_seed
from persona_data.synth_persona import SynthPersonaDataset
from rich.console import Console
from rich.table import Table

from persona_vectors.artifacts import TraitVectorStore
from persona_vectors.attributes import attribute_schema
from persona_vectors.correlations import attribute_association_matrix
from persona_vectors.extraction import MaskStrategy
from persona_vectors.plots.correlations import build_cooccurrence_heatmap
from persona_vectors.steer_generate import generate_steered, steering_coefficient
from persona_vectors.traits import (
    build_trait_direction,
    extract_trait_deltas,
    save_trait_deltas,
)

console = Console()

load_dotenv()
torch.set_grad_enabled(False)
set_seed(1337)

# %% Setting up the model
# Use 9b/70b for remote (production), 2b for local testing.
REMOTE = False
MODEL_NAME = "meta-llama/Llama-3.1-70B-Instruct" if REMOTE else "google/gemma-2-2b-it"

print(f"Loading {MODEL_NAME}...")
model = StandardizedTransformer(MODEL_NAME)
NUM_LAYERS = model.num_layers
D_MODEL = model.hidden_size

# Build all trait directions at one fixed mid-stack layer so their cosines are
# comparable (each layer is a different residual space). The steering notebooks
# show a mid layer steers generation best; AUC across layers is ~flat.
TRAIT_LAYER = NUM_LAYERS // 2
MASK_STRATEGY = MaskStrategy.PERSONA_MEAN  # description-level (flavor A)

model_table = Table(title="Model Config")
model_table.add_column("Property", style="cyan")
model_table.add_column("Value", style="magenta")
model_table.add_row("Model", MODEL_NAME)
model_table.add_row("Layers", str(NUM_LAYERS))
model_table.add_row("Trait layer", str(TRAIT_LAYER))
console.print(model_table)

# %% Load dataset, select personas, and list the binary attributes
N_TRAIN = 1  # PERSONA_MEAN ignores the question, so one QA only builds the prompt
dataset = SynthPersonaDataset(sample_size=8)

# (persona, qa) runs, exactly like notebook_extract; drop personas with no QA.
runs = [
    (persona, dataset.train_test_split(persona.id, n_train=N_TRAIN)[0])
    for persona in dataset
]
runs = [(p, qa) for p, qa in runs if qa]

binary_attrs = [
    name for name, info in attribute_schema(dataset).items() if info.get("kind") == "binary"
]

dataset_table = Table(title="Dataset")
dataset_table.add_column("Property", style="cyan")
dataset_table.add_column("Value", style="magenta")
dataset_table.add_row("Personas with QA", str(len(runs)))
dataset_table.add_row("Binary attributes", ", ".join(binary_attrs))
console.print(dataset_table)

# %% Extract a trait vector per binary attribute
# `verbose=True` on the first attribute prints the contrasted sentence for every
# persona (the minimal-pair diff) plus, once, the averaged token region. Each
# trait vector is saved locally (safetensors + metadata) under
# artifacts/trait_vectors/<model>/<mask>/<variant>/ and reloadable with
# `persona_vectors.traits.load_trait_direction`.
store = TraitVectorStore(MODEL_NAME, mask_strategy=MASK_STRATEGY)
directions: dict[str, dict] = {}
for i, attr in enumerate(binary_attrs):
    console.rule(f"trait: {attr}")
    deltas = extract_trait_deltas(
        model,
        dataset,
        attr,
        runs,
        variant="templated",
        mask_strategy=MASK_STRATEGY,
        remote=REMOTE,
        verbose=(i == 0),
    )
    save_trait_deltas(store, deltas, mask_strategy=MASK_STRATEGY)
    directions[attr] = build_trait_direction(deltas, candidate_layers=[TRAIT_LAYER])

trait_table = Table(title=f"Trait directions @ layer {TRAIT_LAYER}")
for col in ("attribute", "+ (positive)", "auc", "gap_norm", "n"):
    trait_table.add_column(col)
for attr, info in directions.items():
    trait_table.add_row(
        attr,
        str(info["positive"]),
        f"{info['auc']:.3f}",
        f"{info['gap_norm']:.2f}",
        str(info["n_personas"]),
    )
console.print(trait_table)

# %% Trait-cosine matrix (how aligned are the deconfounded directions?)
labels = list(directions)
U = np.stack([directions[a]["unit_direction"].numpy() for a in labels])
cos = np.abs(U @ U.T)  # |cosine|, in [0, 1] to compare against co-occurrence
np.fill_diagonal(cos, 1.0)
build_cooccurrence_heatmap(
    labels,
    cos,
    title=f"Trait-direction similarity (|cosine| @ layer {TRAIT_LAYER})",
    filename="trait_cosine.html",
    show=True,
)

# %% Co-occurrence matrix over the same binary attributes (the baseline)
co_labels, co_matrix = attribute_association_matrix(dataset, attributes=labels)
build_cooccurrence_heatmap(
    co_labels,
    co_matrix,
    title="Attribute co-occurrence (Cramér's V)",
    filename="trait_cooccurrence.html",
    show=True,
)

# %% The "delta": trait-cosine minus co-occurrence
# Near zero where directions track co-occurrence; strongly negative where two
# attributes co-occur yet the trait directions stay orthogonal (deconfounded).
diff = cos - co_matrix
fig = go.Figure(
    go.Heatmap(
        z=diff,
        x=labels,
        y=labels,
        zmin=-1.0,
        zmax=1.0,
        colorscale="RdBu",
        zmid=0.0,
        texttemplate="%{z:.2f}",
        colorbar=dict(title="|cos| − V"),
    )
)
fig.update_layout(
    title="Trait-cosine − co-occurrence",
    template="plotly_white",
    width=700,
    height=640,
)
fig.update_yaxes(autorange="reversed")
fig.show()

# %% Steering sanity check
# Steer toward one binary trait on a neutral human prompt (uniform coefficient,
# swept - / 0 / +) and read off whether the model adopts the trait.
ATTR = "born_in_us"
info = directions[ATTR]
PERSONA_SYS = (
    "You are a human being having a casual conversation. Stay in character and "
    "answer in the first person as a real person. Never say you are an AI."
)
PROMPT = "Tell me about where you were born and where you grew up."

out = generate_steered(
    model,
    PROMPT,
    info["layer"],
    info["unit_direction"],
    [0.0, steering_coefficient(info, 4.0), steering_coefficient(info, -4.0)],
    system=PERSONA_SYS,
    max_new_tokens=120,
    remote=REMOTE,
)
print(f"{ATTR}: + = {info['positive']!r}")
for factor, text in out.items():
    print(f"\n==== coeff={factor:+.2f} ====\n{text}")
