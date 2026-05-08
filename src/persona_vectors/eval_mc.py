"""
Evaluate steering vectors on constrained multiple-choice QA.

For each persona and each alpha scale:
  - Apply per-layer steering vectors during a forward pass.
  - Extract next-token logits after the last prompt token (before the answer).
  - Restrict to the 5 MC choice tokens (A–E) and compute a softmax distribution.
  - Aggregate P(correct choice) across all shared-scope questions.

Train/test split:
  Steering vectors are derived from FRQ activations (scope='individual').
  Evaluation uses scope='shared' MCQs — held out from extraction entirely.

All forward passes for a persona are batched into a single NDIF session to
avoid per-trace job submission overhead.

Produces two plots:
  1. P(correct) vs alpha — mean ± std across questions, one line per persona.
  2. Per-question probability boxplot at alpha=1×.
"""

import nnsight
import torch
from dotenv import load_dotenv
from nnterp import StandardizedTransformer
from persona_data.environment import set_seed
from persona_data.prompts import format_mc_question
from persona_data.synth_persona import SynthPersonaDataset, QAPair
from rich.console import Console
from rich.table import Table
import plotly.graph_objects as go

from persona_vectors.artifacts import HFActivationStore
from persona_vectors.plots import save_plot_html
from persona_vectors.steering import compute_cross_persona_steering_vectors

load_dotenv()
torch.set_grad_enabled(False)
set_seed(1337)

console = Console()

REMOTE = True
MODEL_NAME = "google/gemma-2-9b-it" if REMOTE else "google/gemma-2-2b-it"
HF_REPO = "implicit-personalization/synth-persona-vectors"
MASK_STRATEGY = "answer_mean"
STEER_LAYER = 32
ALPHA_SCALES = [-1.0, 0.0, 0.5, 1.0, 2.0]
N_EVAL_QUESTIONS = None  # None = use all (~475 shared MCQs)

# ── Load model ────────────────────────────────────────────────────────────────

print(f"Loading {MODEL_NAME}...")
model = StandardizedTransformer(MODEL_NAME)

tokenizer = model.tokenizer

CHOICE_LETTERS = ["A", "B", "C", "D", "E"]
choice_token_ids = [
    tokenizer(letter, add_special_tokens=False).input_ids[0]
    for letter in CHOICE_LETTERS
]
choice_token_ids_tensor = torch.tensor(choice_token_ids)

model_table = Table(title="Model Config")
model_table.add_column("Property", style="cyan")
model_table.add_column("Value", style="magenta")
model_table.add_row("Model", MODEL_NAME)
model_table.add_row("Layers", str(model.num_layers))
model_table.add_row("Hidden Size", str(model.hidden_size))
model_table.add_row("Choice token IDs", str(choice_token_ids))
console.print(model_table)

# ── Load dataset ──────────────────────────────────────────────────────────────

dataset = SynthPersonaDataset()
console.print(f"Dataset: {len(dataset)} personas")

# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_prompt_no_answer(qa: QAPair) -> torch.Tensor:
    """Return tokenized prompt up to (but not including) the answer token."""
    messages = [{"role": "user", "content": format_mc_question(qa)}]
    prompt_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return tokenizer(
        prompt_str, return_tensors="pt", add_special_tokens=False
    ).input_ids[0]

def _subset_mean(probs: list[float], indices: list[int]) -> float:
    if not indices:
        return float("nan")
    return sum(probs[i] for i in indices) / len(indices)


def eval_persona(
    qa_pairs: list[QAPair],
    steering_vectors: torch.Tensor,
    suggested_alphas: torch.Tensor,
    layer_indices: list[int],
) -> dict[float, list[float]]:
    """Run all alpha scales × questions in one NDIF session. Returns {scale: [P(correct)]}."""
    input_ids_list = [_format_prompt_no_answer(qa) for qa in qa_pairs]
    n_qa = len(qa_pairs)

    with torch.no_grad(), model.session(remote=REMOTE):
        all_logits = nnsight.save([])
        for scale in ALPHA_SCALES:
            for input_ids in input_ids_list:
                with model.trace(input_ids.unsqueeze(0)):
                    if scale != 0.0:
                        for i, layer_idx in enumerate(layer_indices):
                            h = model.layers_output[layer_idx]
                            sv_layer = steering_vectors[i].to(dtype=h.dtype, device=h.device)
                            alpha = scale * suggested_alphas[i].item()
                            model.layers_output[layer_idx][:] = h + alpha * sv_layer
                    saved = nnsight.save(
                        model.output.logits[0, -1, choice_token_ids_tensor]
                    )
                all_logits.append(saved)

    results: dict[float, list[float]] = {}
    for scale_idx, scale in enumerate(ALPHA_SCALES):
        probs_list: list[float] = []
        for qa_idx, qa in enumerate(qa_pairs):
            logits_5 = all_logits[scale_idx * n_qa + qa_idx]
            probs = torch.softmax(logits_5.float(), dim=0)
            correct_idx = (
                qa.correct_choice_index if qa.correct_choice_index is not None else 0
            )
            probs_list.append(probs[correct_idx].item())
        results[scale] = probs_list

    return results


# ── Load pre-computed vectors from HuggingFace ────────────────────────────────

hf_store = HFActivationStore(
    repo_id=HF_REPO,
    model_name=MODEL_NAME,
    mask_strategy=MASK_STRATEGY,
)

persona_ids = hf_store.list_personas(["biography"])
console.print(f"Found {len(persona_ids)} personas in HF dataset")

sv_dicts = compute_cross_persona_steering_vectors(
    all_persona_ids=persona_ids,
    model_name=MODEL_NAME,
    mask_strategy=MASK_STRATEGY,
    method="mean",
    center=True,
    verbose=False,
    store=hf_store,
)

# ── Main evaluation loop ──────────────────────────────────────────────────────

all_results: dict[str, dict[float, list[float]]] = {}

for persona in dataset:
    if persona.id not in sv_dicts:
        continue

    sv_dict = sv_dicts[persona.id]
    layer_indices = sv_dict["layer_indices"]

    if STEER_LAYER not in layer_indices:
        console.print(f"[yellow]Layer {STEER_LAYER} not available for {persona.name}[/]")
        continue

    li = layer_indices.index(STEER_LAYER)
    layer_indices = [STEER_LAYER]
    steering_vectors = sv_dict["steering_vectors"][li].unsqueeze(0)
    suggested_alphas = sv_dict["suggested_alphas"][li].unsqueeze(0)
    console.print(f"  Steering layer {STEER_LAYER}, suggested_alpha={suggested_alphas[0].item():.4f}")

    _, qa_pairs = dataset.train_test_split(persona.id)
    if N_EVAL_QUESTIONS is not None:
        qa_pairs = qa_pairs[:N_EVAL_QUESTIONS]
    if not qa_pairs:
        console.print(f"[yellow]No shared MCQ questions for {persona.name}[/]")
        continue

    explicit_qs = [q for q in qa_pairs if q.type == "explicit"]
    implicit_qs = [q for q in qa_pairs if q.type == "implicit"]
    console.print(
        f"\n[bold]{persona.name}[/] — {len(qa_pairs)} MCQs "
        f"({len(explicit_qs)} explicit, {len(implicit_qs)} implicit), "
        f"{len(layer_indices)} layers steered"
    )

    persona_results = eval_persona(
        qa_pairs, steering_vectors, suggested_alphas, layer_indices
    )

    explicit_indices = [i for i, q in enumerate(qa_pairs) if q.type == "explicit"]
    implicit_indices = [i for i, q in enumerate(qa_pairs) if q.type == "implicit"]

    table = Table(title=f"MC Eval — {persona.name}")
    table.add_column("Alpha scale", style="cyan", justify="right")
    table.add_column("All (mean)", style="magenta", justify="right")
    table.add_column("Explicit", style="green", justify="right")
    table.add_column("Implicit", style="dim", justify="right")
    for scale in ALPHA_SCALES:
        probs = persona_results[scale]
        mean_all = sum(probs) / len(probs)
        table.add_row(
            f"{scale:+.1f}×",
            f"{mean_all:.3f}",
            f"{_subset_mean(probs, explicit_indices):.3f}",
            f"{_subset_mean(probs, implicit_indices):.3f}",
        )
    console.print(table)

    all_results[persona.name] = persona_results

# ── Plot 1: P(correct) vs alpha scale ─────────────────────────────────────────

fig_curve = go.Figure()
for persona_name, results in all_results.items():
    scales = sorted(results.keys())
    means = [sum(results[s]) / len(results[s]) for s in scales]
    stds = [
        (sum((p - m) ** 2 for p in results[s]) / len(results[s])) ** 0.5
        for s, m in zip(scales, means)
    ]
    fig_curve.add_trace(
        go.Scatter(
            x=scales,
            y=means,
            error_y=dict(type="data", array=stds, visible=True),
            mode="lines+markers",
            name=persona_name,
            hovertemplate="α scale=%{x:.1f}<br>P(correct)=%{y:.3f}<extra>"
            + persona_name
            + "</extra>",
        )
    )

fig_curve.add_hline(
    y=0.2, line_dash="dot", line_color="gray", annotation_text="chance (5 choices)"
)
fig_curve.update_layout(
    title="P(correct choice) vs steering alpha — multi-layer",
    xaxis_title="Alpha scale (× suggested_alpha per layer)",
    yaxis_title="P(correct choice)",
    yaxis=dict(range=[0, 1]),
    template="plotly_white",
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02),
)
out = save_plot_html(fig_curve, "mc_eval_alpha_curve_multilayer")
console.print(f"Plot saved → [cyan]{out}[/]")

# ── Plot 2: Per-question boxplot at alpha=1× ──────────────────────────────────

if all_results and 1.0 in next(iter(all_results.values())):
    fig_box = go.Figure()
    for persona_name, results in all_results.items():
        probs = results.get(1.0, [])
        fig_box.add_trace(
            go.Box(
                y=probs,
                name=persona_name,
                boxpoints="all",
                jitter=0.3,
                pointpos=-1.8,
                hovertemplate="P(correct)=%{y:.3f}<extra>" + persona_name + "</extra>",
            )
        )
    fig_box.add_hline(y=0.2, line_dash="dot", line_color="gray", annotation_text="chance")
    fig_box.update_layout(
        title="Per-question P(correct) at α=1× — multi-layer",
        yaxis_title="P(correct choice)",
        yaxis=dict(range=[0, 1]),
        template="plotly_white",
    )
    out = save_plot_html(fig_box, "mc_eval_per_question_boxplot_multilayer")
    console.print(f"Plot saved → [cyan]{out}[/]")

console.print("\n[green]✓ MC evaluation complete[/]")
