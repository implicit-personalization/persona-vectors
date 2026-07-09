#!/usr/bin/env python

# # Deconfounded trait vectors from minimal-pair attribute swaps
#
# Mirrors `notebook_extract.py`, but instead of one vector per persona we build
# one vector per **ordered attribute**: swap only that attribute on each persona
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
import torch
from dotenv import load_dotenv
from nnterp import StandardizedTransformer
from persona_data.environment import set_seed
from persona_data.synth_persona import BASELINE_PERSONA_ID, SynthPersonaDataset
from rich.console import Console
from rich.table import Table

from persona_vectors.artifacts import TraitVectorStore
from persona_vectors.attributes import attribute_schema
from persona_vectors.correlations import (
    attribute_association_matrix,
    matrix_permutation_test,
    matrix_spearman,
    rank_delta_matrix,
)
from persona_vectors.extraction import MaskStrategy
from persona_vectors.plots.correlations import build_cooccurrence_heatmap
from persona_vectors.steering import PERSONA_SYS, generate_steered, steering_coefficient
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
# Use 9B for remote/report reproduction, 2B for quick local testing.
REMOTE = False
MODEL_NAME = "google/gemma-2-9b-it" if REMOTE else "google/gemma-2-2b-it"
# Number of personas to process per forward pass. batch_size=1 is the original
# sequential path. Larger values right-pad inputs within each chunk and run one
# GPU forward pass per chunk; set to fit VRAM (e.g. 4–8 on an 80 GB A100).
BATCH_SIZE = 2

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

# %% Load dataset, select personas, and list the ordered attributes
N_TRAIN = 1  # PERSONA_MEAN ignores the question, so one QA only builds the prompt
N_PERSONAS = 100 if REMOTE else 2
dataset = SynthPersonaDataset(sample_size=N_PERSONAS)

# (persona, qa) runs, exactly like notebook_extract; drop personas with no QA.
runs = [
    (persona, dataset.train_test_split(persona.id, n_train=N_TRAIN)[0])
    for persona in dataset
    if persona.id != BASELINE_PERSONA_ID
]
runs = [(p, qa) for p, qa in runs if qa]

ordered_attrs = [
    name
    for name, info in attribute_schema(dataset).items()
    if info.get("kind") in {"binary", "ordinal", "numeric"}
]

dataset_table = Table(title="Dataset")
dataset_table.add_column("Property", style="cyan")
dataset_table.add_column("Value", style="magenta")
dataset_table.add_row("Personas with QA", str(len(runs)))
dataset_table.add_row("Ordered attributes", ", ".join(ordered_attrs))
console.print(dataset_table)

# %% Extract a trait vector per ordered attribute
# `verbose=True` on the first attribute prints the contrasted sentence for every
# persona (the minimal-pair diff) plus, once, the averaged token region. Each
# trait vector is saved locally (safetensors + metadata) under
# artifacts/trait_vectors/<model>/<mask>/<variant>/ and reloadable with
# `persona_vectors.traits.load_trait_direction`.
store = TraitVectorStore(MODEL_NAME, mask_strategy=MASK_STRATEGY)
directions: dict[str, dict] = {}
auc_cis: dict[str, tuple[float, float]] = {}

for i, attr in enumerate(ordered_attrs):
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
        batch_size=BATCH_SIZE,
    )

    save_trait_deltas(store, deltas, mask_strategy=MASK_STRATEGY)
    directions[attr] = build_trait_direction(deltas, candidate_layers=[TRAIT_LAYER])
    auc_cis[attr] = deltas.auc_ci(TRAIT_LAYER)

trait_table = Table(title=f"Trait directions @ layer {TRAIT_LAYER}")
for col in ("attribute", "+ (positive)", "in-sample AUC", "bootstrap 95% interval", "gap_norm", "n"):
    trait_table.add_column(col)
for attr, info in directions.items():
    lo, hi = auc_cis[attr]
    trait_table.add_row(
        attr,
        str(info["positive"]),
        f"{info['auc']:.3f}",
        f"[{lo:.3f}, {hi:.3f}]",
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

# %% Co-occurrence matrix over the same ordered attributes (the baseline)
co_labels, co_matrix = attribute_association_matrix(dataset, attributes=labels)
build_cooccurrence_heatmap(
    co_labels,
    co_matrix,
    title="Attribute co-occurrence (Cramér's V)",
    filename="trait_cooccurrence.html",
    show=True,
)

# %% Rank-percentile comparison: trait geometry minus co-occurrence
# Raw |cos| - V is not a meaningful metric difference because cosine and
# Cramér's V use different scales. Instead, rank both off-diagonal matrices and
# plot percentile-rank(|cos|) - percentile-rank(V). Negative cells are pairs that
# co-occur more strongly than their trait directions align; positive cells are
# representation-near despite lower dataset co-occurrence.
rank_delta = rank_delta_matrix(cos, co_matrix)
build_cooccurrence_heatmap(
    labels,
    rank_delta,
    title="Trait geometry rank − co-occurrence rank",
    diverging=True,
    colorbar_title="rank Δ",
    show=True,
)

# %% Rank agreement between geometry and co-occurrence
# Spearman over off-diagonal pairs is the descriptive effect size. The matrix
# permutation p-value is safer than treating all pairs as independent because
# each attribute appears in many matrix cells.
rho, pval, n_pairs = matrix_spearman(cos, co_matrix)
mantel_rho, mantel_p = matrix_permutation_test(cos, co_matrix, n_perm=4999, seed=1337)
print(
    f"Spearman(|cos|, V) over {n_pairs} attribute pairs: "
    f"rho={rho:.2f} (naive p={pval:.3f}); "
    f"matrix-permutation p={mantel_p:.3f}"
)

# %% Steering sanity check
# Steer toward one binary trait on a neutral human prompt (uniform coefficient,
# swept - / 0 / +) and read off whether the model adopts the trait.
ATTR = "born_in_us"
info = directions[ATTR]
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
