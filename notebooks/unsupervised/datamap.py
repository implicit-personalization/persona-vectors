#!/usr/bin/env python
"""Persona data maps: UMAP density clusters named by attributes.

Two maps share one pipeline so they are directly comparable:
  1. Persona vectors - Llama-3.1-405B answer_mean, templated, L65: ~329 QA-answer
     forwards averaged per persona ("how the model answers as that persona").
  2. Persona-mean representation - Llama-3.1-70B persona_mean, templated, L30:
     one forward pass over the persona text, masked mean over its tokens
     ("what the persona text is").

Pipeline (each map): center the per-dim mean (no L2) -> tight cosine UMAP
(n_neighbors=30, min_dist=0.0) -> HDBSCAN (min_cluster_size=15) -> clusters
auto-named by their over-represented attribute values -> datamapplot
interactive render (+ static PNG/SVG).

UMAP knobs: min_dist=0.0 is required (0.1 already blurs the small real pockets);
n_neighbors=30 is the stable middle (10 shatters, 100 dissolves the pockets);
cosine throughout (euclidean collapses the structure into a sex-only split).
HDBSCAN min_cluster_size=15: 10 shatters the map, >=25 merges the small clusters.

Cluster names are summaries, not membership guarantees: a value enters a name
only if it is the cluster majority and that share beats its dataset base rate by
a z-test (see ``attribute_cluster_labels``). Every point's hover ends with its
cluster's purity vs baseline ("Never married 52% (vs 37% overall)").
"""

# %% Imports
import re
from collections import Counter
from pathlib import Path

import datamapplot
import numpy as np
import torch
import umap
from dotenv import load_dotenv
from persona_data.synth_persona import SynthPersonaDataset
from rich.console import Console
from rich.table import Table
from sklearn.cluster import HDBSCAN

from persona_vectors.analysis import load_persona_vectors
from persona_vectors.artifacts import PersonaVectorStore
from persona_vectors.plots import (
    attribute_cluster_labels,
    cluster_color_map,
    persona_datamap,
)

console = Console()

# %% Config
load_dotenv()
torch.set_grad_enabled(False)

MIN_CLUSTER_SIZE = 15  # 10 shatters the map; >=25 merges the small real clusters
OUT_DIR = Path("artifacts/unsupervised/datamapplot")
persona_dataset = SynthPersonaDataset()

# Hover card attributes (the first three are the headline line).
HOVER_ATTRIBUTES = [
    "sex",
    "age",
    "work_status",
    "born_in_us",
    "detailed_race",
    "marital_status",
    "religion",
    "highest_degree_received",
    "political_views",
    "total_wealth",
]
# Attributes a density cluster may be named by (age uses coarse bands below).
NAMING_ATTRIBUTES = [
    "sex",
    "age",
    "work_status",
    "born_in_us",
    "us_citizenship_status",
    "speak_other_language",
    "highest_degree_received",
    "race",
    "detailed_race",
    "marital_status",
    "religion",
    "political_views",
    "family_income_at_16",
    "total_wealth",
]
# Readable stand-ins for raw values that would make bad cluster names.
SHORT_VALUES = {
    ("born_in_us", "Yes"): "US-born",
    ("born_in_us", "No"): "Foreign-born",
    ("speak_other_language", "Yes"): "Bilingual",
    ("speak_other_language", "No"): "English-only",
    ("us_citizenship_status", "A U.S. citizen"): "Citizen",
    ("us_citizenship_status", "Not a U.S. citizen"): "Non-citizen",
}
# Shared look for both interactive maps. The custom js/css make the search box
# self-explanatory: it matches any attribute value of a persona.
MAP_STYLE = dict(
    darkmode=True,
    cluster_boundary_polygons=True,
    polygon_alpha=0.15,
    noise_color="#666666",
    font_family="Montserrat",
    text_outline_width=6,
    min_fontsize=14,
    max_fontsize=28,
    label_wrap_width=24,
    point_radius_max_pixels=18,
    histogram_n_bins=24,
    histogram_settings={
        "histogram_title": "Age (drag to filter)",
        "histogram_bin_fill_color": "#7aa2f7",
        "histogram_bin_selected_fill_color": "#f7768e",
        "histogram_bin_unselected_fill_color": "#3b4261",
    },
    custom_js=(
        'document.getElementById("text-search").placeholder = '
        '"highlight personas by any attribute: Catholic, Divorced, Bilingual, Graduate...";'
    ),
    custom_css="#text-search { width: 420px; }",
)


# %% Per-source attribute columns
def attribute_display(name: str, persona_ids: list[str]) -> list[str]:
    """Attribute values clipped at clause commas ("Unemployed, laid off, ..."
    -> "Unemployed") without breaking dollar amounts ("Less than $5,000")."""
    values = [str(v) for v in persona_dataset.attribute_values(name, persona_ids)]
    return [
        SHORT_VALUES.get((name, v), re.split(r",\s*(?=[A-Za-z])", v)[0]) for v in values
    ]


def attribute_columns(persona_ids: list[str]):
    """Hover values, naming values (age bucketed into bands) and raw ages."""
    ages = [int(v) for v in persona_dataset.attribute_values("age", persona_ids)]
    age_bands = [
        "18-29" if a < 30 else "30-44" if a < 45 else "45-64" if a < 65 else "65+"
        for a in ages
    ]
    hover_values = {
        name: attribute_display(name, persona_ids) for name in HOVER_ATTRIBUTES
    }
    naming_values = {
        name: attribute_display(name, persona_ids) for name in NAMING_ATTRIBUTES
    }
    naming_values["age"] = age_bands  # raw ages never reach a cluster majority
    return hover_values, naming_values, ages


def cluster_table(title: str, labels: np.ndarray) -> None:
    table = Table(title=title)
    table.add_column("Cluster", style="cyan")
    table.add_column("Personas", justify="right")
    for name, count in Counter(labels).most_common():
        table.add_row(str(name), str(count))
    console.print(table)


# %% Build one data map (interactive HTML + static PNG/SVG)
def build_map(
    samples,
    persona_ids: list[str],
    layer: int,
    title: str,
    stem: str,
):
    """Center -> tight cosine UMAP -> HDBSCAN -> named datamapplot figure."""
    hover_values, naming_values, ages = attribute_columns(persona_ids)

    x = samples.vectors[:, layer, :].float().numpy()
    x -= x.mean(axis=0, keepdims=True)
    coords = umap.UMAP(
        n_components=2, n_neighbors=30, min_dist=0.0, metric="cosine", random_state=0
    ).fit_transform(x)

    cluster_ids = HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE).fit_predict(coords)
    labels, details = attribute_cluster_labels(
        cluster_ids, naming_values, include_details=True
    )
    cluster_table(title, labels)
    colors = cluster_color_map(coords, labels)

    cluster_hover = [
        f"{label} — {details[label]}" if label in details else str(label)
        for label in labels
    ]
    sub_title = (
        "each region = a density cluster, named by what its members share"
        " - hover a point for the exact percentages"
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = persona_datamap(
        coords,
        labels,
        {**hover_values, "cluster": cluster_hover},
        title=title,
        sub_title=sub_title,
        histogram_values=ages,
        label_color_map=colors,
        **MAP_STYLE,
    )
    fig.save(str(OUT_DIR / f"{stem}.html"))

    # Static figure: short labels on the map, full purity lines as a caption.
    static_fig, _ = datamapplot.create_plot(
        coords,
        labels,
        darkmode=True,
        label_color_map=colors,
        label_wrap_width=24,
        label_font_size=14,  # uniform; size-by-cluster makes small ones illegible
        font_family="Montserrat",
        title=title,
        sub_title="density clusters named by what their members share",
        figsize=(14, 14),
    )
    caption = "\n".join(
        f"{name} ({int((labels == name).sum())} personas):  {details[name]}"
        for name, _ in Counter(labels).most_common()
        if name in details
    )
    static_fig.text(
        0.05, -0.005, caption, fontsize=12, color="#cccccc", va="top", linespacing=1.5
    )
    static_fig.savefig(OUT_DIR / f"{stem}.png", dpi=200, bbox_inches="tight")
    static_fig.savefig(OUT_DIR / f"{stem}.svg", bbox_inches="tight")
    return fig


# %% Map 1 - persona vectors (405B answer_mean, templated, L65)
pv_store = PersonaVectorStore(
    "meta-llama/Llama-3.1-405B-Instruct",
    mask_strategy="answer_mean",
    root_dir="artifacts/persona-vectors",
)
pv_ids = pv_store.list_personas(["templated"], include_baseline=False)
pv_samples = load_persona_vectors(pv_store, "templated", persona_ids=pv_ids)
console.print(
    f"[bold]persona vectors[/bold] templated: {tuple(pv_samples.vectors.shape)}"
)
pv_fig = build_map(
    pv_samples,
    pv_ids,
    65,
    "Persona vectors - 405B answer_mean L65",
    "datamap_persona_vectors",
)
pv_fig

# %% Map 2 - persona-mean representation (70B persona_mean, templated, L30)
# One forward pass over the persona text (the templated attribute list), masked
# mean over its tokens. Extracted via:
#   python main.py extract --model meta-llama/Llama-3.1-70B-Instruct \
#       --mask-strategy persona_mean --backend remote --variants templated
pm_store = PersonaVectorStore(
    "meta-llama/Llama-3.1-70B-Instruct",
    mask_strategy="persona_mean",
    root_dir="artifacts/activations",
)
pm_ids = pm_store.list_personas(["templated"])
pm_samples = load_persona_vectors(pm_store, "templated", persona_ids=pm_ids)
console.print(f"[bold]persona-mean[/bold] templated: {tuple(pm_samples.vectors.shape)}")
pm_fig = build_map(
    pm_samples,
    pm_ids,
    30,
    "Persona-mean representation - 70B persona_mean L30",
    "datamap_persona_mean",
)
pm_fig
