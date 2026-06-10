#!/usr/bin/env python

"""Attribute co-occurrence across the persona population.

Cramér's V between every pair of persona attributes: how strongly does knowing
one attribute (say `religion`) tell you another (`religion_at_16`)?

Cramér's V is the standard association measure for categorical variables, and
persona attributes are mostly categories; numeric ones (e.g. `age`) are
quantile-binned into a few buckets first so the same measure applies.
"""

# %% Imports
from dotenv import load_dotenv
from persona_data.synth_persona import SynthPersonaDataset
from rich.console import Console
from rich.table import Table

from persona_vectors.correlations import (
    attribute_association_matrix,
    top_cooccurring_pairs,
)
from persona_vectors.plots import build_cooccurrence_heatmap

console = Console()

# %% Setup
load_dotenv()

dataset = SynthPersonaDataset()
attributes = list(dataset.attribute_names)

summary = Table(title="Persona dataset")
summary.add_column("Property", style="cyan")
summary.add_column("Value", style="magenta")
summary.add_row("Personas", str(len(dataset.persona_ids)))
summary.add_row("Attributes", str(len(attributes)))
console.print(summary)

# %% Co-occurrence matrix (Cramér's V; numeric attributes quantile-binned)
labels, matrix = attribute_association_matrix(dataset, attributes)

# %% Heatmap
build_cooccurrence_heatmap(
    labels,
    matrix,
    title="Attribute co-occurrence (Cramér's V)",
).show()


# %% Strongest co-occurring attribute pairs
# The off-diagonal pairs with the highest association
# the attributes most at risk of confounding each other in a trait direction.
pairs = Table(title="Top co-occurring attribute pairs")
pairs.add_column("Attribute A", style="cyan")
pairs.add_column("Attribute B", style="cyan")
pairs.add_column("Cramér's V", style="magenta", justify="right")
for a, b, v in top_cooccurring_pairs(labels, matrix, k=15):
    pairs.add_row(a, b, f"{v:.3f}")
console.print(pairs)
