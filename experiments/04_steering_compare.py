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
from persona_vectors.steering import compute_steering_vector
from persona_vectors.steering_eval_utils import (
    cached_variant_matches,
    load_existing_rows,
    render_summary,
    run_with_oom_retry,
    select_qa_pairs,
    summarize,
    write_outputs,
)

console = Console()

CONFIGS: tuple[dict[str, object], ...] = (
    {
        "name": "legacy_templated_single_mean_nocenter",
        "layer": 20,
        "all_layers": False,
        "negative_variant": "templated",
        "method": "mean",
        "center": False,
    },
    {
        "name": "baseline_single_mean_center",
        "layer": 20,
        "all_layers": False,
        "negative_variant": "baseline",
        "method": "mean",
        "center": True,
    },
    {
        "name": "baseline_all_layers_mean_center",
        "layer": 20,
        "all_layers": True,
        "negative_variant": "baseline",
        "method": "mean",
        "center": True,
    },
    {
        "name": "baseline_all_layers_pca_center",
        "layer": 20,
        "all_layers": True,
        "negative_variant": "baseline",
        "method": "pca",
        "center": True,
    },
)

REFERENCE_CONDITIONS = ("bare", "templated", "biography")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Run a cache-first steering compare on SynthPersona MC. "
            "Shared activations and shared reference conditions are computed once; "
            "each steering config evaluates only the steered condition."
        )
    )
    ap.add_argument("--model", default="google/gemma-2-9b-it")
    ap.add_argument("--personas", type=int, default=30)
    ap.add_argument("--questions-per-persona", type=int, default=56)
    ap.add_argument(
        "--qa-type",
        choices=["implicit", "explicit"],
        default="implicit",
    )
    ap.add_argument("--alpha-scale", type=float, default=1.0)
    ap.add_argument(
        "--configs",
        default=None,
        help=(
            "Comma-separated config names to run. "
            "Defaults to all built-in configs."
        ),
    )
    ap.add_argument(
        "--remote",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument(
        "--out-dir",
        default=None,
        help=(
            "Optional output directory. Defaults to "
            "artifacts/experiments/steering_compare/<timestamp>__<model>__p<...>__q<...>."
        ),
    )
    return ap.parse_args()


def selected_configs(config_arg: str | None) -> tuple[dict[str, object], ...]:
    if not config_arg:
        return CONFIGS
    requested = [name.strip() for name in config_arg.split(",") if name.strip()]
    if not requested:
        return CONFIGS
    by_name = {str(cfg["name"]): cfg for cfg in CONFIGS}
    missing = [name for name in requested if name not in by_name]
    if missing:
        raise ValueError(f"Unknown steering compare config(s): {missing}")
    return tuple(by_name[name] for name in requested)


def default_output_dir(model_name: str, personas: int, questions: int) -> Path:
    run_id = datetime.now().strftime("%Y%m%dT%H%M%SZ")
    model_dir = model_name.replace("/", "__")
    return (
        Path(os.environ.get("ARTIFACTS_DIR", "artifacts"))
        / "experiments"
        / "steering_compare"
        / f"{run_id}__{model_dir}__p{personas}__q{questions}"
    )


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


def row_metadata(args: argparse.Namespace) -> dict[str, object]:
    return {
        "model": args.model,
        "qa_type": args.qa_type,
        "personas": args.personas,
        "questions_per_persona": args.questions_per_persona,
        "alpha_scale": args.alpha_scale,
        "remote": args.remote,
    }


def write_config_outputs(
    out_dir: Path,
    *,
    reference_rows: list[ChoiceEvalResult],
    steered_rows: list[ChoiceEvalResult],
    metadata: dict[str, object],
    failures: list[dict],
) -> None:
    merged_rows = reference_rows + steered_rows
    summary = summarize(merged_rows)
    write_outputs(
        out_dir,
        merged_rows,
        summary,
        metadata,
        failures,
    )
    write_outputs(
        out_dir,
        steered_rows,
        summarize(steered_rows) if steered_rows else {},
        metadata,
        failures,
        jsonl_name="steered_only.jsonl",
        csv_name="steered_only.csv",
    )


def main() -> None:
    args = parse_args()
    load_dotenv()
    set_seed(args.seed)

    if args.remote and not os.environ.get("NDIF_API_KEY"):
        raise RuntimeError("NDIF_API_KEY is not set. Remote steering compare requires NDIF.")

    out_root = (
        Path(args.out_dir)
        if args.out_dir is not None
        else default_output_dir(args.model, args.personas, args.questions_per_persona)
    )
    out_root.mkdir(parents=True, exist_ok=True)
    reference_dir = out_root / "shared_reference"

    dataset = SynthPersonaDataset()
    personas = list(dataset)[: args.personas]
    model = StandardizedTransformer(args.model)
    store = ActivationStore(args.model)
    active_configs = selected_configs(args.configs)

    reference_rows = load_existing_rows(reference_dir / "per_example.jsonl")
    reference_failures_path = reference_dir / "failures.json"
    if reference_failures_path.exists():
        reference_failures = json.loads(reference_failures_path.read_text())
    else:
        reference_failures: list[dict] = []
    failed_reference_ids = {entry["persona_id"] for entry in reference_failures}

    required_variants = ("baseline", "templated", "biography")

    console.rule("Shared activation cache + reference conditions")
    for persona in personas:
        if persona.id in failed_reference_ids:
            console.print(
                f"[yellow]Skipping reference for {persona.name}: previously recorded failure[/]"
            )
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

        if persona_rows_complete(
            reference_rows,
            persona_id=persona.id,
            expected_qids=expected_qids,
            conditions=REFERENCE_CONDITIONS,
        ):
            console.print(f"[cyan]Skipping shared reference for {persona.name}: already complete[/]")
            continue

        if any(row.persona_id == persona.id for row in reference_rows):
            reference_rows = [row for row in reference_rows if row.persona_id != persona.id]

        missing_variants = [
            variant
            for variant in required_variants
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
                    label=f"reference extraction for {persona.name}",
                )

            persona_rows: list[ChoiceEvalResult] = []
            for qa in qa_pairs:
                persona_rows.append(
                    run_with_oom_retry(
                        lambda qa=qa: evaluate_mc_question(
                            model,
                            persona,
                            qa,
                            "bare",
                            remote=args.remote,
                        ),
                        label=f"bare eval for {persona.name} / {qa.qid}",
                    )
                )
                persona_rows.append(
                    run_with_oom_retry(
                        lambda qa=qa: evaluate_mc_question(
                            model,
                            persona,
                            qa,
                            "templated",
                            remote=args.remote,
                        ),
                        label=f"templated eval for {persona.name} / {qa.qid}",
                    )
                )
                persona_rows.append(
                    run_with_oom_retry(
                        lambda qa=qa: evaluate_mc_question(
                            model,
                            persona,
                            qa,
                            "biography",
                            remote=args.remote,
                        ),
                        label=f"biography eval for {persona.name} / {qa.qid}",
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
            console.print(f"[red]Recorded shared reference failure for {persona.name}; continuing[/]")

        reference_summary = summarize(reference_rows)
        reference_metadata = {
            **row_metadata(args),
            "conditions": list(REFERENCE_CONDITIONS),
            "required_variants": list(required_variants),
        }
        write_outputs(
            reference_dir,
            reference_rows,
            reference_summary,
            reference_metadata,
            reference_failures,
        )
        console.print(f"[green]Checkpointed shared reference after {persona.name}[/]")

    reference_summary = summarize(reference_rows)
    reference_metadata = {
        **row_metadata(args),
        "conditions": list(REFERENCE_CONDITIONS),
        "required_variants": list(required_variants),
    }
    write_outputs(
        reference_dir,
        reference_rows,
        reference_summary,
        reference_metadata,
        reference_failures,
    )
    render_summary(reference_summary, title="Shared Reference Summary")

    console.rule("Steered conditions by config")
    for cfg in active_configs:
        cfg_name = str(cfg["name"])
        console.rule(f"Config: {cfg_name}")
        cfg_dir = out_root / cfg_name
        steered_rows = load_existing_rows(cfg_dir / "steered_only.jsonl")
        cfg_failures_path = cfg_dir / "failures.json"
        if cfg_failures_path.exists():
            cfg_failures = json.loads(cfg_failures_path.read_text())
        else:
            cfg_failures: list[dict] = []
        failed_cfg_ids = {entry["persona_id"] for entry in cfg_failures}

        for persona in personas:
            if persona.id in failed_reference_ids:
                continue
            if persona.id in failed_cfg_ids:
                console.print(
                    f"[yellow]Skipping {cfg_name} for {persona.name}: previously recorded failure[/]"
                )
                continue

            qa_pairs = select_qa_pairs(
                dataset,
                persona.id,
                qa_type=args.qa_type,
                limit=args.questions_per_persona,
            )
            if not qa_pairs:
                continue
            expected_qids = [qa.qid for qa in qa_pairs]

            if persona_rows_complete(
                steered_rows,
                persona_id=persona.id,
                expected_qids=expected_qids,
                conditions=("steered",),
            ):
                console.print(f"[cyan]Skipping {cfg_name} for {persona.name}: already complete[/]")
                continue

            if any(row.persona_id == persona.id for row in steered_rows):
                steered_rows = [row for row in steered_rows if row.persona_id != persona.id]

            try:
                sv_dict = compute_steering_vector(
                    persona_id=persona.id,
                    model_name=args.model,
                    layer_idx=None if bool(cfg["all_layers"]) else int(cfg["layer"]),
                    negative_variant=str(cfg["negative_variant"]),
                    method=str(cfg["method"]),
                    center=bool(cfg["center"]),
                    verbose=False,
                )
                if not sv_dict:
                    raise RuntimeError(f"Failed to compute steering vector for {persona.id}")

                steering_alpha = float(sv_dict["suggested_alpha"]) * args.alpha_scale
                steering_vector = sv_dict["steering_vector"]
                steering_layers = sv_dict.get("layers") or [int(cfg["layer"])]

                persona_rows: list[ChoiceEvalResult] = []
                for qa in qa_pairs:
                    persona_rows.append(
                        run_with_oom_retry(
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
                            label=f"{cfg_name} steered eval for {persona.name} / {qa.qid}",
                        )
                    )
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
                console.print(f"[red]Recorded {cfg_name} failure for {persona.name}; continuing[/]")

            cfg_metadata = {
                **row_metadata(args),
                "config_name": cfg_name,
                "layer": int(cfg["layer"]),
                "all_layers": bool(cfg["all_layers"]),
                "negative_variant": str(cfg["negative_variant"]),
                "method": str(cfg["method"]),
                "center": bool(cfg["center"]),
                "shared_reference_dir": str(reference_dir),
            }
            write_config_outputs(
                cfg_dir,
                reference_rows=reference_rows,
                steered_rows=steered_rows,
                metadata=cfg_metadata,
                failures=cfg_failures,
            )
            console.print(f"[green]Checkpointed {cfg_name} after {persona.name}[/]")

        cfg_metadata = {
            **row_metadata(args),
            "config_name": cfg_name,
            "layer": int(cfg["layer"]),
            "all_layers": bool(cfg["all_layers"]),
            "negative_variant": str(cfg["negative_variant"]),
            "method": str(cfg["method"]),
            "center": bool(cfg["center"]),
            "shared_reference_dir": str(reference_dir),
        }
        write_config_outputs(
            cfg_dir,
            reference_rows=reference_rows,
            steered_rows=steered_rows,
            metadata=cfg_metadata,
            failures=cfg_failures,
        )
        render_summary(
            summarize(reference_rows + steered_rows),
            title=f"Steering Compare Summary — {cfg_name}",
        )

    console.print(f"[green]Saved compare outputs to {out_root}[/]")


if __name__ == "__main__":
    main()
