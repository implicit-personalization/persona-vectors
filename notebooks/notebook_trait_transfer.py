#!/usr/bin/env python

"""Held-out transfer checks for a frozen contrastive trait vector.

This notebook never re-fits the trait direction. It projects persona means from
people outside the contrastive extraction cohort onto the saved direction, then
compares that score with the underlying persona attribute. The template-to-
biography section is a stronger check than the source-template result because
the wording changes while the frozen direction does not.

The default local Llama artifact supports ``born_in_us`` with n=100 contrastive
training personas. Other saved Llama traits are small pilots; raise
``MIN_TRAIN_PERSONAS`` only after extracting them on a full cohort.
"""

# %% Imports
import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from persona_data.synth_persona import SynthPersonaDataset
from rich.console import Console
from rich.progress import track
from rich.table import Table
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

from persona_vectors.artifacts import PersonaVectorStore, TraitVectorStore
from persona_vectors.extraction import MaskStrategy

console = Console()
torch.set_grad_enabled(False)

# %% Setup
MODEL_NAME = "meta-llama/Llama-3.1-70B-Instruct"
ATTRIBUTE = "born_in_us"
TRAIT_VARIANT = "templated"
EVALUATION_VARIANTS = ("templated", "biography")
MIN_TRAIN_PERSONAS = 100

# This is chosen before inspecting held-out scores. It is the layer used for
# the source-template projection plot; the all-layer table remains exploratory.
PLOT_LAYER = 29
RUN_PCA = False  # Exploratory only; it is not evidence for deconfounding.

trait_store = TraitVectorStore(
    MODEL_NAME, mask_strategy=MaskStrategy.PERSONA_MEAN
)
persona_store = PersonaVectorStore(
    MODEL_NAME, mask_strategy=MaskStrategy.PERSONA_MEAN
)

# The exact contrastive train IDs should eventually be persisted in the trait
# manifest. Until then, this local run infers the 100-person cohort from the
# matching answer-mean extraction. The count assertion prevents silent use of
# a mismatched cohort.
cohort_store = PersonaVectorStore(
    MODEL_NAME, mask_strategy=MaskStrategy.ANSWER_MEAN
)
train_ids = cohort_store.list_personas([TRAIT_VARIANT])
trait_meta = trait_store.metadata(ATTRIBUTE, variant=TRAIT_VARIANT)
if len(train_ids) != trait_meta["n_personas"] or len(train_ids) < MIN_TRAIN_PERSONAS:
    raise ValueError(
        "The inferred answer-mean cohort does not match the saved trait vector. "
        "Persist and supply the exact contrastive train persona ids instead."
    )

available_ids = persona_store.list_personas(list(EVALUATION_VARIANTS))
test_ids = [persona_id for persona_id in available_ids if persona_id not in train_ids]
if not test_ids:
    raise ValueError("no held-out persona means remain after excluding the train cohort")

dataset = SynthPersonaDataset(sample_size=1000)
positive = str(trait_meta["value_to"])
labels = np.asarray(
    [str(value) == positive for value in dataset.attribute_values(ATTRIBUTE, test_ids)]
)

summary = Table(title="Frozen Trait Transfer")
summary.add_column("Property", style="cyan")
summary.add_column("Value", style="magenta")
summary.add_row("Model", MODEL_NAME)
summary.add_row("Attribute / positive pole", f"{ATTRIBUTE} / {positive}")
summary.add_row("Frozen contrastive train cohort", str(len(train_ids)))
summary.add_row("Held-out personas", str(len(test_ids)))
summary.add_row("Held-out positive / negative", f"{labels.sum()} / {(~labels).sum()}")
console.print(summary)

# %% Frozen direction and population-difference baseline
trait = trait_store.load(ATTRIBUTE, variant=TRAIT_VARIANT).float()
trait_unit = F.normalize(trait, dim=1)


def load_population_difference() -> torch.Tensor:
    """Natural train-population difference: a deliberately confounded baseline."""
    train_labels = np.asarray(
        [str(value) == positive for value in dataset.attribute_values(ATTRIBUTE, train_ids)]
    )
    train_vectors = torch.stack(
        [persona_store.load(TRAIT_VARIANT, persona_id).float() for persona_id in train_ids]
    )
    return train_vectors[train_labels].mean(0) - train_vectors[~train_labels].mean(0)


population_unit = F.normalize(load_population_difference(), dim=1)


def projection_scores(variant: str, direction: torch.Tensor) -> np.ndarray:
    """Stream all held-out vectors to avoid materializing a 900×80×8192 tensor."""
    scores = []
    for persona_id in track(test_ids, description=f"Projecting {variant}"):
        vector = persona_store.load(variant, persona_id).float()
        scores.append((vector * direction).sum(dim=1).numpy())
    return np.stack(scores)


def layer_metrics(scores: np.ndarray) -> list[dict[str, float | int]]:
    """Held-out AUC and standardized projection gap for every layer."""
    rows = []
    for layer in range(scores.shape[1]):
        neg, pos = scores[~labels, layer], scores[labels, layer]
        pooled_sd = np.sqrt((neg.var(ddof=1) + pos.var(ddof=1)) / 2)
        rows.append(
            {
                "layer": layer,
                "auc": float(roc_auc_score(labels, scores[:, layer])),
                "standardized_gap": float((pos.mean() - neg.mean()) / pooled_sd),
            }
        )
    return rows


# %% Held-out source-template and cross-template transfer
trait_scores = {
    variant: projection_scores(variant, trait_unit) for variant in EVALUATION_VARIANTS
}
population_scores = {
    variant: projection_scores(variant, population_unit)
    for variant in EVALUATION_VARIANTS
}

for variant, scores in trait_scores.items():
    rows = layer_metrics(scores)
    best = max(rows, key=lambda row: row["auc"])
    fixed = rows[PLOT_LAYER]
    console.print(
        f"[bold]{variant}[/bold] frozen-trait: "
        f"fixed L{PLOT_LAYER} AUC={fixed['auc']:.3f}, gap={fixed['standardized_gap']:.2f}; "
        f"exploratory best L{best['layer']} AUC={best['auc']:.3f}, gap={best['standardized_gap']:.2f}"
    )

for variant in EVALUATION_VARIANTS:
    trait_fixed = layer_metrics(trait_scores[variant])[PLOT_LAYER]
    population_fixed = layer_metrics(population_scores[variant])[PLOT_LAYER]
    console.print(
        f"[bold]{variant} baseline comparison[/bold] at fixed L{PLOT_LAYER}: "
        f"contrastive AUC={trait_fixed['auc']:.3f}, "
        f"population-difference AUC={population_fixed['auc']:.3f}."
    )

# %% Projection distributions at the pre-specified layer
fig = go.Figure()
for value, selector, color in [
    (positive, labels, "#2563eb"),
    (trait_meta["value_from"], ~labels, "#f97316"),
]:
    fig.add_trace(
        go.Histogram(
            x=trait_scores[TRAIT_VARIANT][selector, PLOT_LAYER],
            name=str(value),
            opacity=0.65,
            marker_color=color,
            histnorm="probability density",
        )
    )
fig.update_layout(
    barmode="overlay",
    title=f"Held-out templated projection onto frozen {ATTRIBUTE} trait (layer {PLOT_LAYER})",
    xaxis_title="dot(persona mean, frozen unit trait)",
    yaxis_title="density",
)
fig.show()

# %% Optional PCA: exploratory visualization, not a deconfounding test
if RUN_PCA:
    vectors = torch.stack(
        [persona_store.load(TRAIT_VARIANT, persona_id)[PLOT_LAYER] for persona_id in test_ids]
    ).float()
    prepared = F.normalize(vectors - vectors.mean(0, keepdim=True), dim=1).numpy()
    coords = PCA(n_components=2, random_state=0).fit_transform(prepared)
    pca_fig = go.Figure()
    for value, selector, color in [
        (positive, labels, "#2563eb"),
        (trait_meta["value_from"], ~labels, "#f97316"),
    ]:
        pca_fig.add_trace(
            go.Scattergl(
                x=coords[selector, 0],
                y=coords[selector, 1],
                mode="markers",
                name=str(value),
                marker={"color": color, "opacity": 0.65},
            )
        )
    pca_fig.update_layout(
        title=f"Exploratory held-out PCA — {ATTRIBUTE}, templated, layer {PLOT_LAYER}",
        xaxis_title="PC 1",
        yaxis_title="PC 2",
    )
    pca_fig.show()

# %% Interpretation
# - The templated score checks cross-person transfer, but can retain template cues.
# - Biography transfer is the stronger no-new-extraction test because wording changes.
# - The population-difference direction is a baseline, not an alternative trait vector:
#   it may predict the natural label well while still carrying correlated attributes.
# - A causal generalization test still needs remote minimal-pair swaps for held-out
#   personas, using the same PERSONA_MEAN mask as this frozen trait direction.
