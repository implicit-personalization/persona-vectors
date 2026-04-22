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
from persona_vectors.extraction import MaskStrategy, run_extraction
from persona_vectors.steering import compute_steering_vector

console = Console()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Evaluate mean-diff persona steering on the current SynthPersona "
            "multiple-choice benchmark."
        )
    )
    ap.add_argument("--model", default="google/gemma-2-9b-it")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--personas", type=int, default=10)
    ap.add_argument("--questions-per-persona", type=int, default=30)
    ap.add_argument(
        "--qa-type",
        choices=["implicit", "explicit"],
        default="implicit",
    )
    ap.add_argument(
        "--alpha-scale",
        type=float,
        default=1.0,
        help="Multiplier on top of the suggested alpha from the steering vector.",
    )
    ap.add_argument(
        "--remote",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run traces on NDIF remote servers.",
    )
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Optional output directory. Defaults to artifacts/experiments/steering_eval/<timestamp>.",
    )
    return ap.parse_args()


def default_output_dir(model_name: str, qa_type: str, personas: int, questions: int) -> Path:
    run_id = datetime.now().strftime("%Y%m%dT%H%M%SZ")
    model_dir = model_name.replace("/", "__")
    return (
        Path(os.environ.get("ARTIFACTS_DIR", "artifacts"))
        / "experiments"
        / "steering_eval"
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

    for condition in ["templated", "biography", "steered"]:
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
        fieldnames = list(row_dicts[0].keys())
        with (out_dir / "per_example.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(row_dicts)


def render_summary(summary: dict[str, dict[str, float | int]]) -> None:
    table = Table(title="Steering Eval Summary")
    table.add_column("Condition", style="cyan")
    table.add_column("Metric", style="magenta")
    table.add_column("Value", style="green")

    for condition, metrics in summary.items():
        first = True
        for key, value in metrics.items():
            if isinstance(value, float):
                value_str = f"{value:.4f}"
            else:
                value_str = str(value)
            table.add_row(condition if first else "", key, value_str)
            first = False
    console.print(table)


def main() -> None:
    args = parse_args()
    load_dotenv()
    set_seed(args.seed)

    if args.remote and not os.environ.get("NDIF_API_KEY"):
        raise RuntimeError(
            "NDIF_API_KEY is not set. Remote steering eval requires an NDIF API key."
        )

    out_dir = (
        Path(args.out_dir)
        if args.out_dir is not None
        else default_output_dir(
            args.model, args.qa_type, args.personas, args.questions_per_persona
        )
    )

    dataset = SynthPersonaDataset()
    personas = list(dataset)[: args.personas]
    model = StandardizedTransformer(args.model)
    rows: list[ChoiceEvalResult] = []

    for persona in personas:
        qa_pairs = select_qa_pairs(
            dataset,
            persona.id,
            qa_type=args.qa_type,
            limit=args.questions_per_persona,
        )
        if not qa_pairs:
            console.print(f"[yellow]Skipping {persona.name}: no matching QA pairs[/]")
            continue

        run_extraction(
            model=model,
            model_name=args.model,
            persona=persona,
            qa_pairs=qa_pairs,
            variants=("templated", "biography"),
            mask_strategy=MaskStrategy.RESPONSE_MEAN,
            remote=args.remote,
            verbose=False,
        )
        sv_dict = compute_steering_vector(
            persona_id=persona.id,
            model_name=args.model,
            layer_idx=args.layer,
            verbose=False,
        )
        if not sv_dict:
            raise RuntimeError(f"Failed to compute steering vector for {persona.id}")

        steering_alpha = float(sv_dict["suggested_alpha"]) * args.alpha_scale
        steering_vector = sv_dict["steering_vector"]

        for qa in qa_pairs:
            rows.append(
                evaluate_mc_question(
                    model,
                    persona,
                    qa,
                    "bare",
                    remote=args.remote,
                )
            )
            rows.append(
                evaluate_mc_question(
                    model,
                    persona,
                    qa,
                    "templated",
                    remote=args.remote,
                )
            )
            rows.append(
                evaluate_mc_question(
                    model,
                    persona,
                    qa,
                    "biography",
                    remote=args.remote,
                )
            )
            rows.append(
                evaluate_mc_question(
                    model,
                    persona,
                    qa,
                    "steered",
                    remote=args.remote,
                    steering_layer=args.layer,
                    steering_vector=steering_vector,
                    steering_alpha=steering_alpha,
                )
            )

    summary = summarize(rows)
    metadata = {
        "model": args.model,
        "layer": args.layer,
        "qa_type": args.qa_type,
        "personas": args.personas,
        "questions_per_persona": args.questions_per_persona,
        "alpha_scale": args.alpha_scale,
        "remote": args.remote,
    }
    write_outputs(out_dir, rows, summary, metadata)
    render_summary(summary)
    console.print(f"[green]Saved outputs to {out_dir}[/]")


if __name__ == "__main__":
    main()
