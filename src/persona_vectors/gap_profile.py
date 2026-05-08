"""
Visualize per-layer persona gap and pairwise inter-persona vector similarity.
"""

import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from pathlib import Path
from persona_data.environment import set_seed
from persona_data.synth_persona import SynthPersonaDataset
from rich.console import Console
from rich.table import Table
import plotly.graph_objects as go

from persona_vectors.artifacts import list_personas
from persona_vectors.plots import save_plot_html
from persona_vectors.steering import compute_steering_vectors, compute_cross_persona_steering_vectors

REPO_ROOT = Path(__file__).parent.parent.parent
ACTIVATIONS_DIR = REPO_ROOT / "artifacts" / "activations"

# "original"     → biography − templated  (one persona at a time)
# "cross_persona" → biography − mean(other biographies)  (all personas together)
METHOD = "cross_persona"

# Layer to check pairwise cosine similarity at
COSIM_LAYER = 20

load_dotenv()
torch.set_grad_enabled(False)
set_seed(1337)

console = Console()

MODEL_NAME = "google/gemma-2-9b-it"

dataset = SynthPersonaDataset()

gap_fig = go.Figure()
all_sv_dicts: list[tuple[str, dict]] = []

if METHOD == "cross_persona":
    persona_ids = list_personas(ACTIVATIONS_DIR, MODEL_NAME, ["biography"])
    console.print(f"Computing cross-persona vectors for {len(persona_ids)} personas...")
    cross_results = compute_cross_persona_steering_vectors(
        all_persona_ids=persona_ids,
        model_name=MODEL_NAME,
        activations_dir=ACTIVATIONS_DIR,
        mask_strategy="answer_mean",
        method="mean",
        center=True,
        verbose=True,
    )
    # Build a name lookup
    id_to_name = {p.id: p.name for p in dataset}
    for pid, sv_dict in cross_results.items():
        name = id_to_name.get(pid, pid[:8])
        all_sv_dicts.append((name, sv_dict))
        gap_fig.add_trace(go.Scatter(
            x=sv_dict["layer_indices"],
            y=sv_dict["persona_gaps"].tolist(),
            mode="lines", name=name, opacity=0.6,
            hovertemplate=f"{name}<br>layer=%{{x}}<br>gap=%{{y:.2f}}<extra></extra>",
        ))
else:
    for persona in dataset:
        sv_dict = compute_steering_vectors(
            persona_id=persona.id,
            model_name=MODEL_NAME,
            activations_dir=ACTIVATIONS_DIR,
            method="mean",
            center=True,
            verbose=False,
        )
        if not sv_dict:
            continue
        all_sv_dicts.append((persona.name, sv_dict))
        gap_fig.add_trace(go.Scatter(
            x=sv_dict["layer_indices"],
            y=sv_dict["persona_gaps"].tolist(),
            mode="lines", name=persona.name, opacity=0.6,
            hovertemplate=f"{persona.name}<br>layer=%{{x}}<br>gap=%{{y:.2f}}<extra></extra>",
        ))

console.print(f"Loaded {len(all_sv_dicts)} personas")

for layer, label, color in [(20, "layer 20 (old default)", "orange"),
                             (32, "layer 32 (ARENA Gemma 2)", "green")]:
    gap_fig.add_vline(x=layer, line_dash="dot", line_color=color,
                      annotation_text=label, annotation_position="top")

gap_fig.update_layout(
    title="Per-layer persona gap (‖mean(bio_h − tmpl_h)‖₂)",
    xaxis_title="Layer index",
    yaxis_title="Persona gap (L2 norm of mean diff)",
    template="plotly_white",
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02),
)
out = save_plot_html(gap_fig, "persona_gap_profile_by_layer")
console.print(f"Gap profile saved → [cyan]{out}[/]")

# ── Pairwise cosine similarity at COSIM_LAYER ─────────────────────────────────

if len(all_sv_dicts) >= 2:
    # Extract the steering vector at COSIM_LAYER for each persona
    vecs: list[torch.Tensor] = []
    names: list[str] = []
    for name, sv_dict in all_sv_dicts:
        layer_indices = sv_dict["layer_indices"]
        if COSIM_LAYER not in layer_indices:
            continue
        li = layer_indices.index(COSIM_LAYER)
        vecs.append(F.normalize(sv_dict["steering_vectors"][li].float(), dim=0))
        names.append(name)

    if len(vecs) >= 2:
        V = torch.stack(vecs)                          # (n, hidden_dim)
        cosim_matrix = (V @ V.T).cpu()                 # (n, n)

        # Print summary table: mean off-diagonal cosine similarity
        n = len(vecs)
        off_diag = cosim_matrix[~torch.eye(n, dtype=torch.bool)].mean().item()
        table = Table(title=f"Inter-persona cosine similarity @ layer {COSIM_LAYER}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_row("Personas compared", str(n))
        table.add_row("Mean off-diagonal cosim", f"{off_diag:.4f}")
        table.add_row(
            "Interpretation",
            "vectors are generic (not persona-specific)" if off_diag > 0.8
            else "vectors are moderately distinct" if off_diag > 0.4
            else "vectors are persona-specific",
        )
        console.print(table)

        # Heatmap
        cosim_fig = go.Figure(
            go.Heatmap(
                z=cosim_matrix.tolist(),
                x=names, y=names,
                colorscale="RdBu",
                zmid=0,
                zmin=-1, zmax=1,
                hovertemplate="%{y} vs %{x}<br>cosim=%{z:.3f}<extra></extra>",
            )
        )
        cosim_fig.update_layout(
            title=f"Pairwise steering-vector cosine similarity @ layer {COSIM_LAYER}",
            template="plotly_white",
        )
        out2 = save_plot_html(cosim_fig, f"sv_cosim_layer{COSIM_LAYER}")
        console.print(f"Cosim heatmap saved → [cyan]{out2}[/]")
    else:
        console.print(f"[yellow]Need ≥2 personas with layer {COSIM_LAYER} to compute cosim[/]")

console.print("\n[green]✓ Gap profile complete[/]")
