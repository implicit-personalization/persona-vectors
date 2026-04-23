#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
from nnterp import StandardizedTransformer
from persona_data.environment import set_seed
from persona_data.synth_persona import SynthPersonaDataset
from rich.console import Console

from persona_vectors.eval import ChoiceEvalResult, evaluate_mc_questions_batch
from persona_vectors.steering import compute_steering_vector
from persona_vectors.steering_eval_utils import (
    load_existing_rows,
    render_summary,
    run_with_remote_retry,
    select_qa_pairs,
    summarize,
    write_outputs,
)

console = Console()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Fast cross-persona alpha sweep. "
            "Compute bare reference once, then evaluate only steered_own and "
            "steered_cross for each alpha."
        )
    )
    ap.add_argument("--model", default="google/gemma-2-9b-it")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--all-layers", action="store_true")
    ap.add_argument("--personas", type=int, default=3)
    ap.add_argument("--persona-ids", default=None)
    ap.add_argument("--questions-per-persona", type=int, default=20)
    ap.add_argument("--qa-type", choices=["implicit", "explicit"], default="implicit")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--remote", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--negative-variant", choices=["templated", "baseline", "pooled_biography"], default="pooled_biography")
    ap.add_argument("--method", choices=["mean", "pca"], default="mean")
    ap.add_argument("--center", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--alphas",
        default="0.5,1.0,2.0,3.0",
        help="Comma-separated fixed steering alphas to sweep.",
    )
    ap.add_argument(
        "--question-batch-size",
        type=int,
        default=6,
        help="How many MC questions to score per NDIF trace.",
    )
    ap.add_argument("--out-dir", default=None)
    return ap.parse_args()


def default_output_dir(model_name: str, qa_type: str, personas: int, questions: int) -> Path:
    run_id = datetime.now().strftime("%Y%m%dT%H%M%SZ")
    model_dir = model_name.replace("/", "__")
    return (
        Path("artifacts")
        / "experiments"
        / "cross_persona_alpha_sweep_fast"
        / f"{run_id}__{model_dir}__{qa_type}__p{personas}__q{questions}"
    )


def chunked(items: list, chunk_size: int) -> Iterable[list]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


def persona_rows_complete(
    rows: list[ChoiceEvalResult],
    *,
    persona_id: str,
    expected_qids: list[str],
    conditions: tuple[str, ...],
) -> bool:
    persona_rows = [row for row in rows if row.persona_id == persona_id]
    if len(persona_rows) != len(expected_qids) * len(conditions):
        return False
    got = {(row.qid, row.condition) for row in persona_rows}
    want = {(qid, condition) for qid in expected_qids for condition in conditions}
    return got == want


def write_config_outputs(
    out_dir: Path,
    *,
    reference_rows: list[ChoiceEvalResult],
    steered_rows: list[ChoiceEvalResult],
    metadata: dict,
    failures: list[dict],
) -> None:
    merged_rows = reference_rows + steered_rows
    summary = summarize(merged_rows)
    write_outputs(out_dir, merged_rows, summary, metadata, failures)
    render_summary(summary, title=f"Fast alpha sweep summary — alpha={metadata['alpha_override']}")


def main() -> None:
    args = parse_args()
    load_dotenv()
    set_seed(args.seed)

    if args.remote and not os.environ.get("NDIF_API_KEY"):
        raise RuntimeError("NDIF_API_KEY is required for remote runs.")

    alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]
    if not alphas:
        raise ValueError("Need at least one alpha")

    out_root = Path(args.out_dir) if args.out_dir else default_output_dir(
        args.model, args.qa_type, args.personas, args.questions_per_persona
    )
    out_root.mkdir(parents=True, exist_ok=True)
    reference_dir = out_root / "shared_reference"

    dataset = SynthPersonaDataset()
    if args.persona_ids:
        wanted_ids = [item.strip() for item in args.persona_ids.split(",") if item.strip()]
        persona_by_id = {persona.id: persona for persona in dataset}
        personas = [persona_by_id[persona_id] for persona_id in wanted_ids]
    else:
        personas = list(dataset)[: args.personas]
    model = StandardizedTransformer(args.model)

    reference_rows = load_existing_rows(reference_dir / "per_example.jsonl")
    reference_failures_path = reference_dir / "failures.json"
    if reference_failures_path.exists():
        reference_failures = json.loads(reference_failures_path.read_text())
    else:
        reference_failures = []
    failed_reference_ids = {entry["persona_id"] for entry in reference_failures}

    # bare reference once
    for persona in personas:
        if persona.id in failed_reference_ids:
            continue
        qa_pairs = select_qa_pairs(dataset, persona.id, args.qa_type, args.questions_per_persona)
        expected_qids = [qa.qid for qa in qa_pairs]
        if persona_rows_complete(reference_rows, persona_id=persona.id, expected_qids=expected_qids, conditions=("bare",)):
            console.print(f"[cyan]Skipping bare reference for {persona.name}: already complete[/]")
            continue

        if any(row.persona_id == persona.id for row in reference_rows):
            reference_rows = [row for row in reference_rows if row.persona_id != persona.id]

        try:
            persona_rows = []
            for batch_idx, qa_batch in enumerate(
                chunked(qa_pairs, args.question_batch_size), start=1
            ):
                persona_rows.extend(
                    run_with_remote_retry(
                        lambda qa_batch=qa_batch: evaluate_mc_questions_batch(
                            model,
                            persona,
                            qa_batch,
                            "bare",
                            remote=args.remote,
                        ),
                        label=f"bare batch eval for {persona.name} chunk {batch_idx}",
                    )
                )
            reference_rows.extend(persona_rows)
        except Exception as exc:
            reference_failures.append(
                {
                    "persona_id": persona.id,
                    "persona_name": persona.name,
                    "error": str(exc),
                }
            )
            failed_reference_ids.add(persona.id)
            console.print(f"[red]Recorded bare-reference failure for {persona.name}; continuing[/]")

        write_outputs(
            reference_dir,
            reference_rows,
            summarize(reference_rows),
            {
                "model": args.model,
                "qa_type": args.qa_type,
                "personas": len(personas),
                "questions_per_persona": args.questions_per_persona,
                "remote": args.remote,
                "reference_conditions": ["bare"],
            },
            reference_failures,
        )
        console.print(f"[green]Checkpointed bare reference after {persona.name}[/]")

    # compute vectors once
    vector_bank: dict[str, tuple[str, object, object]] = {}
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
            sv_dict.get("layers") or [args.layer],
        )

    ordered_ids = [persona.id for persona in personas]
    cross_source_for: dict[str, str] = {}
    for idx, persona_id in enumerate(ordered_ids):
        cross_source_for[persona_id] = ordered_ids[(idx + 1) % len(ordered_ids)]

    for alpha in alphas:
        safe_alpha = str(alpha).replace(".", "p")
        cfg_dir = out_root / f"alpha_{safe_alpha}"
        steered_rows = load_existing_rows(cfg_dir / "steered_only.jsonl")
        cfg_failures_path = cfg_dir / "failures.json"
        if cfg_failures_path.exists():
            cfg_failures = json.loads(cfg_failures_path.read_text())
        else:
            cfg_failures = []
        failed_cfg_ids = {entry["persona_id"] for entry in cfg_failures}

        console.rule(f"Alpha={alpha}")
        for persona in personas:
            if persona.id in failed_reference_ids or persona.id in failed_cfg_ids:
                continue

            qa_pairs = select_qa_pairs(dataset, persona.id, args.qa_type, args.questions_per_persona)
            expected_qids = [qa.qid for qa in qa_pairs]
            if persona_rows_complete(
                steered_rows,
                persona_id=persona.id,
                expected_qids=expected_qids,
                conditions=("steered_own", "steered_cross"),
            ):
                console.print(f"[cyan]Skipping alpha={alpha} for {persona.name}: already complete[/]")
                continue

            if any(row.persona_id == persona.id for row in steered_rows):
                steered_rows = [row for row in steered_rows if row.persona_id != persona.id]

            _, own_vector, own_layers = vector_bank[persona.id]
            cross_id = cross_source_for[persona.id]
            _, cross_vector, cross_layers = vector_bank[cross_id]

            try:
                own_rows = []
                for batch_idx, qa_batch in enumerate(
                    chunked(qa_pairs, args.question_batch_size), start=1
                ):
                    own_rows.extend(
                        run_with_remote_retry(
                            lambda qa_batch=qa_batch: evaluate_mc_questions_batch(
                                model,
                                persona,
                                qa_batch,
                                "steered",
                                remote=args.remote,
                                steering_layer=own_layers if args.all_layers else args.layer,
                                steering_vector=own_vector,
                                steering_alpha=alpha,
                            ),
                            label=f"{persona.name} steered_own chunk {batch_idx}",
                        )
                    )
                for row in own_rows:
                    row.condition = "steered_own"

                cross_rows = []
                for batch_idx, qa_batch in enumerate(
                    chunked(qa_pairs, args.question_batch_size), start=1
                ):
                    cross_rows.extend(
                        run_with_remote_retry(
                            lambda qa_batch=qa_batch: evaluate_mc_questions_batch(
                                model,
                                persona,
                                qa_batch,
                                "steered",
                                remote=args.remote,
                                steering_layer=cross_layers if args.all_layers else args.layer,
                                steering_vector=cross_vector,
                                steering_alpha=alpha,
                            ),
                            label=f"{persona.name} steered_cross chunk {batch_idx}",
                        )
                    )
                for row in cross_rows:
                    row.condition = "steered_cross"

                persona_rows = own_rows + cross_rows
                steered_rows.extend(persona_rows)
            except Exception as exc:
                cfg_failures.append(
                    {
                        "persona_id": persona.id,
                        "persona_name": persona.name,
                        "error": str(exc),
                    }
                )
                failed_cfg_ids.add(persona.id)
                console.print(f"[red]Recorded alpha={alpha} failure for {persona.name}; continuing[/]")

            metadata = {
                "model": args.model,
                "qa_type": args.qa_type,
                "personas": len(personas),
                "questions_per_persona": args.questions_per_persona,
                "remote": args.remote,
                "negative_variant": args.negative_variant,
                "method": args.method,
                "center": args.center,
                "all_layers": args.all_layers,
                "layer": args.layer,
                "alpha_override": alpha,
                "cross_source_for": cross_source_for,
                "question_batch_size": args.question_batch_size,
            }
            write_config_outputs(
                cfg_dir,
                reference_rows=reference_rows,
                steered_rows=steered_rows,
                metadata=metadata,
                failures=cfg_failures,
            )
            console.print(f"[green]Checkpointed alpha={alpha} after {persona.name}[/]")

    console.print(f"[green]Saved fast alpha sweep outputs to {out_root}[/]")


if __name__ == "__main__":
    main()
