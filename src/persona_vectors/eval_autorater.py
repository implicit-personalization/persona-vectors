"""
Autorater evaluation for persona steering.

Generates free-text responses via NDIF at each alpha scale, then scores each
response against the persona biography using TogetherAI (Llama 3.1 70B).
Produces a plot of mean autorater score vs alpha across personas.
"""

import os

import torch
from dotenv import load_dotenv
from nnterp import StandardizedTransformer
from openai import OpenAI
from persona_data.environment import set_seed
from persona_data.prompts import format_prompt
from persona_data.synth_persona import QAPair, SynthPersonaDataset
from rich.console import Console
from rich.table import Table
import plotly.graph_objects as go

from persona_vectors.artifacts import HFActivationStore
from persona_vectors.plots import save_plot_html
from persona_vectors.steering import compute_cross_persona_steering_vectors

load_dotenv()
set_seed(1337)
console = Console()

# ── Config ────────────────────────────────────────────────────────────────────

REMOTE = True
MODEL_NAME = "google/gemma-2-9b-it" if REMOTE else "google/gemma-2-2b-it"
HF_REPO = "implicit-personalization/synth-persona-vectors"
MASK_STRATEGY = "answer_mean"
STEER_LAYER = 32
ALPHA_SCALES = [-1.0, 0.0, 0.5, 1.0, 2.0]

N_GEN_QUESTIONS = 5    # FRQ questions to generate responses for per persona
MAX_NEW_TOKENS = 150   # max response length

AUTORATER_MODEL = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"

together = OpenAI(
    api_key=os.environ["TOGETHER_KEY"],
    base_url="https://api.together.xyz/v1",
)

# ── Load model and dataset ────────────────────────────────────────────────────

print(f"Loading {MODEL_NAME}...")
model = StandardizedTransformer(MODEL_NAME)
tokenizer = model.tokenizer

dataset = SynthPersonaDataset()
console.print(f"Dataset: {len(dataset)} personas")

# ── Load HF vectors and compute cross-persona steering vectors ────────────────

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

# ── Helpers ───────────────────────────────────────────────────────────────────

def _tokenize_question(question: str) -> torch.Tensor:
    """Tokenize a bare user question for generation (no system prompt)."""
    messages = [{"role": "user", "content": question}]
    prompt_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return tokenizer(
        prompt_str, return_tensors="pt", add_special_tokens=False
    ).input_ids[0]


def generate_responses(
    qa_pairs: list[QAPair],
    steering_vector: torch.Tensor,
    suggested_alpha: float,
    layer_idx: int,
) -> dict[float, list[str]]:
    """Generate steered free-text responses for all alpha scales × questions in one NDIF session."""
    input_ids_list = [_tokenize_question(qa.question) for qa in qa_pairs]

    saved: list = []  # flat list: [scale0_q0, scale0_q1, ..., scale1_q0, ...]

    with torch.no_grad(), model.session(remote=REMOTE):
        for scale in ALPHA_SCALES:
            alpha = scale * suggested_alpha
            for input_ids in input_ids_list:
                with model.trace(input_ids.unsqueeze(0), max_new_tokens=MAX_NEW_TOKENS) as tracer:
                    if alpha != 0.0:
                        h = model.layers_output[layer_idx]
                        sv = steering_vector.to(dtype=h.dtype, device=h.device)
                        model.layers_output[layer_idx][:] = h + alpha * sv
                    sequences = model.output.sequences.save()
                    tracer.stop()
                saved.append(sequences)

    prompt_len = input_ids_list[0].shape[0]
    results: dict[float, list[str]] = {}
    for scale_idx, scale in enumerate(ALPHA_SCALES):
        decoded = []
        for q_idx in range(len(qa_pairs)):
            seq = saved[scale_idx * len(qa_pairs) + q_idx]
            gen_tokens = seq[0, prompt_len:]
            text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            decoded.append(text)
        results[scale] = decoded
    return results


def score_response(persona_bio: str, question: str, response: str) -> float | None:
    """Score a response against a persona bio. Returns 0-100 or None on parse failure."""
    prompt = (
        f"Here is a description of a person:\n{persona_bio}\n\n"
        f'They were asked: "{question}"\n\n'
        f'They responded: "{response}"\n\n'
        "On a scale of 0-100, how well does this response reflect the personality, "
        "values, knowledge, and background of the person described above? "
        "Consider whether the content, tone, and perspective are consistent with "
        "the persona description.\n\n"
        "Reply with only a single integer from 0 to 100."
    )
    resp = together.chat.completions.create(
        model=AUTORATER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10,
        temperature=0.0,
    )
    try:
        return float(resp.choices[0].message.content.strip().split()[0])
    except (ValueError, IndexError, AttributeError):
        return None


def _mean(values: list[float | None]) -> float:
    valid = [v for v in values if v is not None]
    return sum(valid) / len(valid) if valid else float("nan")


# ── Main eval loop ────────────────────────────────────────────────────────────

all_results: dict[str, dict[float, float]] = {}

for persona in dataset:
    if persona.id not in sv_dicts:
        continue

    sv_dict = sv_dicts[persona.id]
    layer_indices = sv_dict["layer_indices"]
    if STEER_LAYER not in layer_indices:
        console.print(f"[yellow]Layer {STEER_LAYER} not available for {persona.name}[/]")
        continue

    li = layer_indices.index(STEER_LAYER)
    steering_vector = sv_dict["steering_vectors"][li]       # (hidden_dim,)
    suggested_alpha = sv_dict["suggested_alphas"][li].item()

    train_qa, _ = dataset.train_test_split(persona.id)
    gen_qa = train_qa[:N_GEN_QUESTIONS]
    if not gen_qa:
        console.print(f"[yellow]No train FRQs for {persona.name}[/]")
        continue

    persona_bio = format_prompt(persona, "biography")

    console.print(
        f"\n[bold]{persona.name}[/] — generating {len(gen_qa)} responses "
        f"× {len(ALPHA_SCALES)} alpha values"
    )

    responses_by_scale = generate_responses(
        gen_qa, steering_vector, suggested_alpha, STEER_LAYER
    )

    scores_by_scale: dict[float, float] = {}
    for scale in ALPHA_SCALES:
        raw_scores = [
            score_response(persona_bio, qa.question, response)
            for qa, response in zip(gen_qa, responses_by_scale[scale])
        ]
        scores_by_scale[scale] = _mean(raw_scores)

    table = Table(title=f"Autorater — {persona.name}")
    table.add_column("Alpha scale", style="cyan", justify="right")
    table.add_column("Mean score (0-100)", style="magenta", justify="right")
    for scale in ALPHA_SCALES:
        table.add_row(f"{scale:+.1f}×", f"{scores_by_scale[scale]:.1f}")
    console.print(table)

    all_results[persona.name] = scores_by_scale

# ── Plot ──────────────────────────────────────────────────────────────────────

fig = go.Figure()
for persona_name, scores in all_results.items():
    scales = sorted(scores.keys())
    fig.add_trace(go.Scatter(
        x=scales,
        y=[scores[s] for s in scales],
        mode="lines+markers",
        name=persona_name,
        hovertemplate=f"{persona_name}<br>α=%{{x:.1f}}×<br>score=%{{y:.1f}}<extra></extra>",
    ))

fig.add_hline(y=50, line_dash="dot", line_color="gray", annotation_text="neutral")
fig.update_layout(
    title="Autorater score vs steering alpha (cross-persona, biography)",
    xaxis_title="Alpha scale (× suggested_alpha)",
    yaxis_title="Mean autorater score (0-100)",
    yaxis=dict(range=[0, 100]),
    template="plotly_white",
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02),
)
out = save_plot_html(fig, "autorater_score_vs_alpha")
console.print(f"\nPlot saved → [cyan]{out}[/]")
console.print("\n[green]✓ Autorater eval complete[/]")
