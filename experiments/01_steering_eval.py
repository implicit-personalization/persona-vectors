#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from nnterp import StandardizedTransformer
from persona_data.environment import set_seed
from persona_data.synth_persona import SynthPersonaDataset
from rich.console import Console

from persona_vectors.artifacts import ActivationStore
from persona_vectors.eval import ChoiceEvalResult, evaluate_mc_question
from persona_vectors.extraction import MaskStrategy, run_extraction
from persona_vectors.steering_eval_utils import (
    cached_variant_matches,
    load_existing_rows,
    render_summary,
    run_with_oom_retry,
    select_qa_pairs,
    summarize,
    write_outputs,
)
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
    ap.add_argument(
        "--all-layers",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Inject per-layer steering vectors at every layer instead of only one layer.",
    )
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
        "--negative-variant",
        choices=["templated", "baseline", "pooled_biography"],
        default="baseline",
        help="Negative prompt variant used to compute the steering vector.",
    )
    ap.add_argument(
        "--method",
        choices=["mean", "pca"],
        default="mean",
        help="How to aggregate per-question activation diffs into a steering vector.",
    )
    ap.add_argument(
        "--center",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Center each saved activation vector before computing steering diffs.",
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
    store = ActivationStore(args.model)
    rows: list[ChoiceEvalResult] = load_existing_rows(out_dir)
    failures_path = out_dir / "failures.json"
    if failures_path.exists():
        failures = json.loads(failures_path.read_text())
    else:
        failures: list[dict] = []
    failed_persona_ids = {entry["persona_id"] for entry in failures}

    for persona in personas:
        if persona.id in failed_persona_ids:
            console.print(f"[yellow]Skipping {persona.name}: previously recorded failure[/]")
            continue

        qa_pairs = select_qa_pairs(
            dataset,
            persona.id,
            qa_type=args.qa_type,
            limit=args.questions_per_persona,
        )
        if not qa_pairs:
            console.print(f"[yellow]Skipping {persona.name}: no matching QA pairs[/]")
            continue

        expected_qids = [qa.qid for qa in qa_pairs]
        expected_rows = len(qa_pairs) * 4
        persona_rows = [row for row in rows if row.persona_id == persona.id]
        if len(persona_rows) == expected_rows and {
            row.qid for row in persona_rows
        } == set(expected_qids):
            console.print(f"[cyan]Skipping {persona.name}: completed rows already exist[/]")
            continue
        if persona_rows:
            rows = [row for row in rows if row.persona_id != persona.id]

        missing_variants = [
            variant
            for variant in {"templated", "biography", args.negative_variant}
            if not cached_variant_matches(store, variant, persona.id, expected_qids)
        ]
        try:
            if missing_variants:
                run_with_oom_retry(
                    lambda: run_extraction(
                        model=model,
                        model_name=args.model,
                        persona=persona,
                        qa_pairs=qa_pairs,
                        variants=tuple(missing_variants),
                        mask_strategy=MaskStrategy.RESPONSE_MEAN,
                        remote=args.remote,
                        verbose=False,
                        chunk_size=1 if args.remote else None,
                    ),
                    label=f"extraction for {persona.name}",
                )

            sv_dict = compute_steering_vector(
                persona_id=persona.id,
                model_name=args.model,
                layer_idx=None if args.all_layers else args.layer,
                negative_variant=args.negative_variant,
                method=args.method,
                center=args.center,
                verbose=False,
            )
            if not sv_dict:
                raise RuntimeError(f"Failed to compute steering vector for {persona.id}")

            steering_alpha = float(sv_dict["suggested_alpha"]) * args.alpha_scale
            steering_vector = sv_dict["steering_vector"]
            steering_layers = sv_dict.get("layers") or [args.layer]

            for qa in qa_pairs:
                rows.append(run_with_oom_retry(
                    lambda qa=qa: evaluate_mc_question(
                        model,
                        persona,
                        qa,
                        "bare",
                        remote=args.remote,
                    ),
                    label=f"bare eval for {persona.name} / {qa.qid}",
                ))
                rows.append(run_with_oom_retry(
                    lambda qa=qa: evaluate_mc_question(
                        model,
                        persona,
                        qa,
                        "templated",
                        remote=args.remote,
                    ),
                    label=f"templated eval for {persona.name} / {qa.qid}",
                ))
                rows.append(run_with_oom_retry(
                    lambda qa=qa: evaluate_mc_question(
                        model,
                        persona,
                        qa,
                        "biography",
                        remote=args.remote,
                    ),
                    label=f"biography eval for {persona.name} / {qa.qid}",
                ))
                rows.append(run_with_oom_retry(
                    lambda qa=qa: evaluate_mc_question(
                        model,
                        persona,
                        qa,
                        "steered",
                        remote=args.remote,
                        steering_layer=steering_layers,
                        steering_vector=steering_vector,
                        steering_alpha=steering_alpha,
                    ),
                    label=f"steered eval for {persona.name} / {qa.qid}",
                ))
        except Exception as exc:
            failures.append(
                {
                    "persona_id": persona.id,
                    "persona_name": persona.name,
                    "error": str(exc),
                }
            )
            failed_persona_ids.add(persona.id)
            partial_summary = summarize(rows)
            partial_metadata = {
                "model": args.model,
                "layer": args.layer,
                "all_layers": args.all_layers,
                "qa_type": args.qa_type,
                "personas": args.personas,
                "questions_per_persona": args.questions_per_persona,
                "alpha_scale": args.alpha_scale,
                "negative_variant": args.negative_variant,
                "method": args.method,
                "center": args.center,
                "remote": args.remote,
            }
            write_outputs(out_dir, rows, partial_summary, partial_metadata, failures)
            console.print(f"[red]Recorded failure for {persona.name}; continuing[/]")
            continue

        partial_summary = summarize(rows)
        partial_metadata = {
            "model": args.model,
            "layer": args.layer,
            "all_layers": args.all_layers,
            "qa_type": args.qa_type,
            "personas": args.personas,
            "questions_per_persona": args.questions_per_persona,
            "alpha_scale": args.alpha_scale,
            "negative_variant": args.negative_variant,
            "method": args.method,
            "center": args.center,
            "remote": args.remote,
        }
        write_outputs(out_dir, rows, partial_summary, partial_metadata, failures)
        console.print(f"[green]Checkpointed after {persona.name}[/]")

    summary = summarize(rows)
    metadata = {
        "model": args.model,
        "layer": args.layer,
        "all_layers": args.all_layers,
        "qa_type": args.qa_type,
        "personas": args.personas,
        "questions_per_persona": args.questions_per_persona,
        "alpha_scale": args.alpha_scale,
        "negative_variant": args.negative_variant,
        "method": args.method,
        "center": args.center,
        "remote": args.remote,
    }
    write_outputs(out_dir, rows, summary, metadata, failures)
    render_summary(summary, title="Steering Eval Summary")
    console.print(f"[green]Saved outputs to {out_dir}[/]")


if __name__ == "__main__":
    main()
