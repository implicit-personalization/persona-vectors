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

from persona_vectors.artifacts import ActivationStore
from persona_vectors.eval import ChoiceEvalResult, evaluate_mc_questions_batch
from persona_vectors.extraction import MaskStrategy, run_extraction
from persona_vectors.mc_prompt_contract import MC_PROMPT_CONTRACT_VERSION
from persona_vectors.steering import compute_steering_vector
from persona_vectors.steering_eval_utils import (
    cached_variant_matches,
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
            "Re-extract activations under the corrected synth-persona MC prompt "
            "contract, then rerun pooled-bio cross-persona steering sanity checks."
        )
    )
    ap.add_argument("--model", default="google/gemma-2-9b-it")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--all-layers", action="store_true")
    ap.add_argument("--personas", type=int, default=3)
    ap.add_argument("--persona-ids", default=None)
    ap.add_argument(
        "--vector-personas",
        type=int,
        default=None,
        help="How many personas to use for the pooled-negative extraction pool. Defaults to --personas.",
    )
    ap.add_argument(
        "--vector-persona-ids",
        default=None,
        help="Optional comma-separated persona ids for the extraction/vector pool. Defaults to the eval personas.",
    )
    ap.add_argument("--questions-per-persona", type=int, default=20)
    ap.add_argument("--qa-type", choices=["implicit", "explicit"], default="implicit")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--remote", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--negative-variant",
        choices=["templated", "baseline", "pooled_biography"],
        default="pooled_biography",
    )
    ap.add_argument("--method", choices=["mean", "pca"], default="mean")
    ap.add_argument("--center", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--mask-strategy",
        choices=[strategy.value for strategy in MaskStrategy],
        default=MaskStrategy.RESPONSE_MEAN.value,
        help="Which token span to average when extracting activations.",
    )
    ap.add_argument(
        "--alphas",
        default="1.0,2.0",
        help="Comma-separated fixed steering alphas to sweep.",
    )
    ap.add_argument(
        "--question-batch-size",
        type=int,
        default=5,
        help="How many MC questions to score per NDIF trace.",
    )
    ap.add_argument(
        "--extraction-batch-size",
        type=int,
        default=None,
        help="How many activation-extraction prompts to run per NDIF trace. Defaults to --question-batch-size.",
    )
    ap.add_argument(
        "--activation-root",
        default=None,
        help="Optional activation cache root. Defaults to <out-dir>/activations.",
    )
    ap.add_argument(
        "--skip-extraction-failures",
        action="store_true",
        help="Record extraction failures and continue with the successfully cached vector pool.",
    )
    ap.add_argument("--out-dir", default=None)
    return ap.parse_args()


def default_output_dir(model_name: str, qa_type: str, personas: int, questions: int) -> Path:
    run_id = datetime.now().strftime("%Y%m%dT%H%M%SZ")
    model_dir = model_name.replace("/", "__")
    return (
        Path("artifacts")
        / "experiments"
        / "cross_persona_contract_rerun"
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


def extraction_variants_for(negative_variant: str) -> tuple[str, ...]:
    if negative_variant == "pooled_biography":
        return ("biography",)
    return ("biography", negative_variant)


def write_extraction_summary(
    out_root: Path,
    *,
    store: ActivationStore,
    extraction_variants: tuple[str, ...],
    vector_personas: list,
    extraction_failures: list[dict],
) -> dict:
    failed_ids = {entry["persona_id"] for entry in extraction_failures}
    cached_ids_by_variant: dict[str, list[str]] = {}
    for variant in extraction_variants:
        cached_ids: list[str] = []
        for persona in vector_personas:
            try:
                store.load_metadata(variant, persona.id)
            except FileNotFoundError:
                continue
            cached_ids.append(persona.id)
        cached_ids_by_variant[variant] = cached_ids

    summary = {
        "planned_vector_personas": len(vector_personas),
        "extraction_failure_count": len(extraction_failures),
        "extraction_failure_personas": [
            {
                "persona_id": entry["persona_id"],
                "persona_name": entry["persona_name"],
                "error": entry["error"],
            }
            for entry in extraction_failures
        ],
        "cached_counts_by_variant": {
            variant: len(cached_ids) for variant, cached_ids in cached_ids_by_variant.items()
        },
        "dropped_persona_ids": sorted(failed_ids),
    }
    (out_root / "extraction_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


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
    write_outputs(
        out_dir,
        merged_rows,
        summary,
        metadata,
        failures,
        jsonl_name="per_example.jsonl",
        csv_name="per_example.csv",
    )
    with (out_dir / "steered_only.jsonl").open("w") as f:
        for row in steered_rows:
            f.write(json.dumps(row.to_dict()) + "\n")
    render_summary(summary, title=f"Contract rerun summary — alpha={metadata['alpha_override']}")


def main() -> None:
    args = parse_args()
    load_dotenv()
    set_seed(args.seed)

    if args.remote and not os.environ.get("NDIF_API_KEY"):
        raise RuntimeError("NDIF_API_KEY is required for remote runs.")

    alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]
    if not alphas:
        raise ValueError("Need at least one alpha")
    mask_strategy = MaskStrategy(args.mask_strategy)

    out_root = Path(args.out_dir) if args.out_dir else default_output_dir(
        args.model, args.qa_type, args.personas, args.questions_per_persona
    )
    out_root.mkdir(parents=True, exist_ok=True)
    os.environ["ARTIFACTS_DIR"] = str(out_root)
    activation_root = (
        Path(args.activation_root) if args.activation_root else out_root / "activations"
    )
    reference_dir = out_root / "shared_reference"

    dataset = SynthPersonaDataset()
    persona_by_id = {persona.id: persona for persona in dataset}
    ordered_personas = list(dataset)
    if args.persona_ids:
        wanted_ids = [item.strip() for item in args.persona_ids.split(",") if item.strip()]
        personas = [persona_by_id[persona_id] for persona_id in wanted_ids]
    else:
        personas = ordered_personas[: args.personas]

    if args.vector_persona_ids:
        vector_ids = [item.strip() for item in args.vector_persona_ids.split(",") if item.strip()]
        vector_personas = [persona_by_id[persona_id] for persona_id in vector_ids]
    elif args.vector_personas is not None:
        vector_personas = ordered_personas[: args.vector_personas]
    else:
        vector_personas = personas
    model = StandardizedTransformer(args.model)
    store = ActivationStore(args.model, activation_root)

    extraction_variants = extraction_variants_for(args.negative_variant)
    extraction_failures_path = out_root / "extraction_failures.json"
    if extraction_failures_path.exists():
        extraction_failures = json.loads(extraction_failures_path.read_text())
    else:
        extraction_failures = []
    failed_extraction_ids = {entry["persona_id"] for entry in extraction_failures}
    for persona in vector_personas:
        if args.skip_extraction_failures and persona.id in failed_extraction_ids:
            console.print(
                f"[yellow]Skipping extraction for {persona.name}: recorded previous extraction failure[/]"
            )
            continue
        qa_pairs = select_qa_pairs(dataset, persona.id, args.qa_type, args.questions_per_persona)
        expected_qids = [qa.qid for qa in qa_pairs]
        missing = [
            variant
            for variant in extraction_variants
            if not cached_variant_matches(
                store,
                variant,
                persona.id,
                expected_qids,
                expected_prompt_contract_version=MC_PROMPT_CONTRACT_VERSION,
                expected_mask_strategy=mask_strategy.value,
            )
        ]
        if not missing:
            console.print(
                f"[cyan]Skipping extraction for {persona.name}: {','.join(extraction_variants)} already cached with corrected contract[/]"
            )
            continue
        try:
            run_with_remote_retry(
                lambda persona=persona, qa_pairs=qa_pairs, missing=missing: run_extraction(
                    model,
                    args.model,
                    persona,
                    qa_pairs,
                    tuple(missing),
                    mask_strategy=mask_strategy,
                    remote=args.remote,
                    chunk_size=args.extraction_batch_size or args.question_batch_size,
                ),
                label=f"corrected-contract extraction for {persona.name}",
                retries=5,
                sleep_seconds=10,
            )
        except Exception as exc:
            extraction_failures.append(
                {
                    "persona_id": persona.id,
                    "persona_name": persona.name,
                    "missing_variants": missing,
                    "error": str(exc),
                }
            )
            failed_extraction_ids.add(persona.id)
            extraction_failures_path.write_text(json.dumps(extraction_failures, indent=2))
            if args.skip_extraction_failures:
                console.print(
                    f"[yellow]Dropping {persona.name} from vector pool after extraction failure[/]"
                )
                continue
            raise RuntimeError(
                f"Corrected-contract extraction failed for {persona.name} on variants {missing}"
            ) from exc

    extraction_failures_path.write_text(json.dumps(extraction_failures, indent=2))
    extraction_summary = write_extraction_summary(
        out_root,
        store=store,
        extraction_variants=extraction_variants,
        vector_personas=vector_personas,
        extraction_failures=extraction_failures,
    )

    vector_bank: dict[str, dict] = {}
    for persona in personas:
        sv_dict = compute_steering_vector(
            persona_id=persona.id,
            model_name=args.model,
            layer_idx=None if args.all_layers else args.layer,
            activations_dir=activation_root,
            negative_variant=args.negative_variant,
            method=args.method,
            center=args.center,
            verbose=False,
        )
        vector_bank[persona.id] = {
            "name": persona.name,
            "vector": sv_dict["steering_vector"],
            "layers": sv_dict.get("layers") or [args.layer],
            "suggested_alpha": float(sv_dict["suggested_alpha"]),
            "metadata": {
                key: value
                for key, value in sv_dict.items()
                if key not in {"steering_vector"}
            },
        }
    (out_root / "vector_bank_metadata.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "negative_variant": args.negative_variant,
                "method": args.method,
                "center": args.center,
                "all_layers": args.all_layers,
                "eval_personas": [persona.id for persona in personas],
                "vector_pool_personas": [persona.id for persona in vector_personas],
                "activation_root": str(activation_root),
                "extraction_summary": extraction_summary,
                "question_batch_size": args.question_batch_size,
                "extraction_batch_size": args.extraction_batch_size
                or args.question_batch_size,
                "prompt_contract_version": MC_PROMPT_CONTRACT_VERSION,
                "mask_strategy": mask_strategy.value,
                "vectors": {
                    persona_id: payload["metadata"] for persona_id, payload in vector_bank.items()
                },
            },
            indent=2,
        )
    )

    ordered_ids = [persona.id for persona in personas]
    cross_source_for: dict[str, str] = {}
    for idx, persona_id in enumerate(ordered_ids):
        cross_source_for[persona_id] = ordered_ids[(idx + 1) % len(ordered_ids)]

    reference_rows = load_existing_rows(reference_dir / "per_example.jsonl")
    reference_failures_path = reference_dir / "failures.json"
    if reference_failures_path.exists():
        reference_failures = json.loads(reference_failures_path.read_text())
    else:
        reference_failures = []
    failed_reference_ids = {entry["persona_id"] for entry in reference_failures}

    for persona in personas:
        if persona.id in failed_reference_ids:
            continue
        qa_pairs = select_qa_pairs(dataset, persona.id, args.qa_type, args.questions_per_persona)
        expected_qids = [qa.qid for qa in qa_pairs]
        if persona_rows_complete(
            reference_rows,
            persona_id=persona.id,
            expected_qids=expected_qids,
            conditions=("bare",),
        ):
            console.print(f"[cyan]Skipping bare reference for {persona.name}: already complete[/]")
            continue
        if any(row.persona_id == persona.id for row in reference_rows):
            reference_rows = [row for row in reference_rows if row.persona_id != persona.id]
        try:
            persona_rows: list[ChoiceEvalResult] = []
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
                "vector_pool_personas": len(vector_personas),
                "questions_per_persona": args.questions_per_persona,
                "activation_root": str(activation_root),
                "question_batch_size": args.question_batch_size,
                "extraction_batch_size": args.extraction_batch_size
                or args.question_batch_size,
                "remote": args.remote,
                "reference_conditions": ["bare"],
                "prompt_contract_version": MC_PROMPT_CONTRACT_VERSION,
                "mask_strategy": mask_strategy.value,
                "negative_variant": args.negative_variant,
                "method": args.method,
                "center": args.center,
                "all_layers": args.all_layers,
                "extraction_variants": list(extraction_variants),
            },
            reference_failures,
        )
        console.print(f"[green]Checkpointed bare reference after {persona.name}[/]")

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

        console.rule(f"Corrected-contract alpha={alpha}")
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

            own_payload = vector_bank[persona.id]
            cross_id = cross_source_for[persona.id]
            cross_payload = vector_bank[cross_id]

            try:
                persona_rows: list[ChoiceEvalResult] = []
                for batch_idx, qa_batch in enumerate(
                    chunked(qa_pairs, args.question_batch_size), start=1
                ):
                    own_batch = run_with_remote_retry(
                        lambda qa_batch=qa_batch: evaluate_mc_questions_batch(
                            model,
                            persona,
                            qa_batch,
                            "steered",
                            remote=args.remote,
                            steering_layer=own_payload["layers"],
                            steering_vector=own_payload["vector"],
                            steering_alpha=alpha,
                        ),
                        label=f"own batch eval for {persona.name} alpha={alpha} chunk {batch_idx}",
                    )
                    for row in own_batch:
                        row.condition = "steered_own"
                    persona_rows.extend(own_batch)

                    cross_batch = run_with_remote_retry(
                        lambda qa_batch=qa_batch: evaluate_mc_questions_batch(
                            model,
                            persona,
                            qa_batch,
                            "steered",
                            remote=args.remote,
                            steering_layer=cross_payload["layers"],
                            steering_vector=cross_payload["vector"],
                            steering_alpha=alpha,
                        ),
                        label=f"cross batch eval for {persona.name} alpha={alpha} chunk {batch_idx}",
                    )
                    for row in cross_batch:
                        row.condition = "steered_cross"
                    persona_rows.extend(cross_batch)
                steered_rows.extend(persona_rows)
            except Exception as exc:
                cfg_failures.append(
                    {
                        "persona_id": persona.id,
                        "persona_name": persona.name,
                        "alpha": alpha,
                        "cross_source_id": cross_id,
                        "cross_source_name": cross_payload["name"],
                        "error": str(exc),
                    }
                )
                failed_cfg_ids.add(persona.id)
                console.print(
                    f"[red]Recorded alpha={alpha} failure for {persona.name}; continuing[/]"
                )

            write_config_outputs(
                cfg_dir,
                reference_rows=reference_rows,
                steered_rows=steered_rows,
                metadata={
                    "model": args.model,
                    "qa_type": args.qa_type,
                    "personas": len(personas),
                    "vector_pool_personas": len(vector_personas),
                    "questions_per_persona": args.questions_per_persona,
                    "activation_root": str(activation_root),
                    "question_batch_size": args.question_batch_size,
                    "extraction_batch_size": args.extraction_batch_size
                    or args.question_batch_size,
                    "remote": args.remote,
                    "negative_variant": args.negative_variant,
                    "method": args.method,
                    "center": args.center,
                    "all_layers": args.all_layers,
                    "alpha_override": alpha,
                    "prompt_contract_version": MC_PROMPT_CONTRACT_VERSION,
                    "mask_strategy": mask_strategy.value,
                    "cross_source_for": cross_source_for,
                    "extraction_variants": list(extraction_variants),
                },
                failures=cfg_failures,
            )
            console.print(f"[green]Checkpointed alpha={alpha} after {persona.name}[/]")


if __name__ == "__main__":
    main()
