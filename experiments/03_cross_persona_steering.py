#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from nnterp import StandardizedTransformer
from persona_data.environment import set_seed
from persona_data.synth_persona import SynthPersonaDataset
from rich.console import Console
from rich.table import Table

from persona_vectors.eval import ChoiceEvalResult, evaluate_mc_question
from persona_vectors.steering import compute_steering_vector
from persona_vectors.steering_eval_utils import run_with_remote_retry

console = Console()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Compare own-vector steering against cross-persona steering on SynthPersona MC."
    )
    ap.add_argument("--model", default="google/gemma-2-2b-it")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument(
        "--all-layers",
        action="store_true",
        help="Apply one steering vector per layer instead of a single layer-20 vector.",
    )
    ap.add_argument("--personas", type=int, default=10)
    ap.add_argument(
        "--persona-ids",
        default=None,
        help="Optional comma-separated persona ids. If set, overrides --personas ordering.",
    )
    ap.add_argument("--questions-per-persona", type=int, default=7)
    ap.add_argument(
        "--qa-type",
        choices=["implicit", "explicit"],
        default="implicit",
    )
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument(
        "--remote",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Optional output directory. Defaults to artifacts/experiments/cross_persona_steering/<timestamp>.",
    )
    ap.add_argument(
        "--negative-variant",
        choices=["templated", "baseline", "pooled_biography"],
        default="baseline",
    )
    ap.add_argument(
        "--method",
        choices=["mean", "pca"],
        default="mean",
    )
    ap.add_argument(
        "--center",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    ap.add_argument(
        "--alpha-override",
        type=float,
        default=None,
        help="If set, use the same fixed steering alpha for own and cross vectors.",
    )
    return ap.parse_args()


def default_output_dir(model_name: str, qa_type: str, personas: int, questions: int) -> Path:
    run_id = datetime.now().strftime("%Y%m%dT%H%M%SZ")
    model_dir = model_name.replace("/", "__")
    return (
        Path("artifacts")
        / "experiments"
        / "cross_persona_steering"
        / f"{run_id}__{model_dir}__{qa_type}__p{personas}__q{questions}"
    )


def select_qa_pairs(dataset: SynthPersonaDataset, persona_id: str, qa_type: str, limit: int):
    qa_pairs = [
        qa
        for qa in dataset.get_qa(persona_id)
        if qa.type == qa_type and qa.answer_format == "choice"
    ]
    return qa_pairs[:limit]


def summarize(rows: list[ChoiceEvalResult]) -> dict[str, dict[str, float | int]]:
    by_condition: dict[str, list[ChoiceEvalResult]] = defaultdict(list)
    for row in rows:
        by_condition[row.condition].append(row)

    summary: dict[str, dict[str, float | int]] = {}
    for condition, cond_rows in by_condition.items():
        n = len(cond_rows)
        summary[condition] = {
            "n_examples": n,
            "accuracy": sum(int(r.correct) for r in cond_rows) / n if n else 0.0,
            "mean_gold_prob": sum(r.gold_prob for r in cond_rows) / n if n else 0.0,
            "mean_gold_logprob": (
                sum(r.gold_logprob for r in cond_rows) / n if n else 0.0
            ),
            "mean_margin_vs_best_other": (
                sum(r.margin_vs_best_other for r in cond_rows) / n if n else 0.0
            ),
        }

    grouped: dict[tuple[str, str], dict[str, ChoiceEvalResult]] = defaultdict(dict)
    for row in rows:
        grouped[(row.persona_id, row.qid)][row.condition] = row

    for condition in ["templated", "biography", "steered_own", "steered_cross"]:
        deltas: list[float] = []
        flip_to_gold = 0
        flip_away_from_gold = 0
        any_flip = 0
        comparable = 0
        for pair in grouped.values():
            if "bare" not in pair or condition not in pair:
                continue
            comparable += 1
            bare = pair["bare"]
            other = pair[condition]
            deltas.append(other.gold_prob - bare.gold_prob)
            if other.predicted_letter != bare.predicted_letter:
                any_flip += 1
            if (not bare.correct) and other.correct:
                flip_to_gold += 1
            if bare.correct and (not other.correct):
                flip_away_from_gold += 1

        if comparable:
            summary[f"{condition}_vs_bare"] = {
                "n_examples": comparable,
                "mean_delta_gold_prob": sum(deltas) / comparable,
                "flip_to_gold": flip_to_gold,
                "flip_away_from_gold": flip_away_from_gold,
                "any_flip": any_flip,
            }

    return summary


def write_outputs(
    out_dir: Path,
    rows: list[ChoiceEvalResult],
    summary: dict[str, dict[str, float | int]],
    metadata: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    row_dicts = [row.to_dict() for row in rows]
    with (out_dir / "per_example.jsonl").open("w") as f:
        for row in row_dicts:
            f.write(json.dumps(row) + "\n")
    with (out_dir / "summary.json").open("w") as f:
        json.dump({"metadata": metadata, "summary": summary}, f, indent=2)
    if row_dicts:
        with (out_dir / "per_example.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row_dicts[0].keys()))
            writer.writeheader()
            writer.writerows(row_dicts)


def render_summary(summary: dict[str, dict[str, float | int]]) -> None:
    table = Table(title="Cross-persona steering summary")
    table.add_column("Condition", style="cyan")
    table.add_column("Metric", style="magenta")
    table.add_column("Value", style="green")
    for condition, metrics in summary.items():
        first = True
        for key, value in metrics.items():
            value_str = f"{value:.4f}" if isinstance(value, float) else str(value)
            table.add_row(condition if first else "", key, value_str)
            first = False
    console.print(table)


def main() -> None:
    args = parse_args()
    load_dotenv()
    set_seed(args.seed)

    if args.remote and not os.environ.get("NDIF_API_KEY"):
        raise RuntimeError("NDIF_API_KEY is required for remote runs.")

    out_dir = Path(args.out_dir) if args.out_dir else default_output_dir(
        args.model, args.qa_type, args.personas, args.questions_per_persona
    )

    dataset = SynthPersonaDataset()
    if args.persona_ids:
        wanted_ids = [item.strip() for item in args.persona_ids.split(",") if item.strip()]
        persona_by_id = {persona.id: persona for persona in dataset}
        personas = [persona_by_id[persona_id] for persona_id in wanted_ids]
    else:
        personas = list(dataset)[: args.personas]
    model = StandardizedTransformer(args.model)

    vector_bank: dict[str, tuple[str, object, float, object]] = {}
    for persona in personas:
        sv_dict = compute_steering_vector(
            persona_id=persona.id,
            model_name=args.model,
            layer_idx=None if args.all_layers else args.layer,
            negative_variant=args.negative_variant,
            method=args.method,
            center=args.center,
            verbose=False,
        )
        vector_bank[persona.id] = (
            persona.name,
            sv_dict["steering_vector"],
            float(sv_dict["suggested_alpha"]),
            sv_dict.get("layers") or [args.layer],
        )

    ordered_ids = [persona.id for persona in personas]
    cross_source_for: dict[str, str] = {}
    for idx, persona_id in enumerate(ordered_ids):
        cross_source_for[persona_id] = ordered_ids[(idx + 1) % len(ordered_ids)]

    rows: list[ChoiceEvalResult] = []
    for persona in personas:
        qa_pairs = select_qa_pairs(dataset, persona.id, args.qa_type, args.questions_per_persona)
        own_name, own_vector, own_alpha, own_layers = vector_bank[persona.id]
        cross_id = cross_source_for[persona.id]
        cross_name, cross_vector, cross_alpha, cross_layers = vector_bank[cross_id]
        if args.alpha_override is not None:
            own_alpha = float(args.alpha_override)
            cross_alpha = float(args.alpha_override)

        for qa in qa_pairs:
            rows.append(
                run_with_remote_retry(
                    lambda: evaluate_mc_question(model, persona, qa, "bare", remote=args.remote),
                    label=f"{persona.name} {qa.qid} bare",
                )
            )
            rows.append(
                run_with_remote_retry(
                    lambda: evaluate_mc_question(model, persona, qa, "templated", remote=args.remote),
                    label=f"{persona.name} {qa.qid} templated",
                )
            )
            rows.append(
                run_with_remote_retry(
                    lambda: evaluate_mc_question(model, persona, qa, "biography", remote=args.remote),
                    label=f"{persona.name} {qa.qid} biography",
                )
            )
            own_row = run_with_remote_retry(
                lambda: evaluate_mc_question(
                    model,
                    persona,
                    qa,
                    "steered",
                    remote=args.remote,
                    steering_layer=own_layers if args.all_layers else args.layer,
                    steering_vector=own_vector,
                    steering_alpha=own_alpha,
                ),
                label=f"{persona.name} {qa.qid} steered_own",
            )
            own_row.condition = "steered_own"
            rows.append(own_row)
            cross_row = run_with_remote_retry(
                lambda: evaluate_mc_question(
                    model,
                    persona,
                    qa,
                    "steered",
                    remote=args.remote,
                    steering_layer=cross_layers if args.all_layers else args.layer,
                    steering_vector=cross_vector,
                    steering_alpha=cross_alpha,
                ),
                label=f"{persona.name} {qa.qid} steered_cross",
            )
            cross_row.condition = "steered_cross"
            rows.append(cross_row)

        summary = summarize(rows)
        checkpoint_metadata = {
            "model": args.model,
            "layer": args.layer,
            "all_layers": args.all_layers,
            "negative_variant": args.negative_variant,
            "method": args.method,
            "center": args.center,
            "qa_type": args.qa_type,
            "personas": args.personas,
            "questions_per_persona": args.questions_per_persona,
            "remote": args.remote,
            "cross_source_for": cross_source_for,
            "alpha_override": args.alpha_override,
            "checkpoint_after_persona_id": persona.id,
            "checkpoint_after_persona_name": persona.name,
        }
        write_outputs(out_dir, rows, summary, checkpoint_metadata)
        console.print(f"[cyan]Checkpointed after {persona.name}[/]")

    summary = summarize(rows)
    metadata = {
        "model": args.model,
        "layer": args.layer,
        "all_layers": args.all_layers,
        "negative_variant": args.negative_variant,
        "method": args.method,
        "center": args.center,
        "qa_type": args.qa_type,
        "personas": args.personas,
        "questions_per_persona": args.questions_per_persona,
        "remote": args.remote,
        "cross_source_for": cross_source_for,
        "alpha_override": args.alpha_override,
    }
    write_outputs(out_dir, rows, summary, metadata)
    render_summary(summary)
    console.print(f"[green]Saved outputs to {out_dir}[/]")


if __name__ == "__main__":
    main()
