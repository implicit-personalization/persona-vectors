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
from persona_vectors.static_construction import (
    build_contrast_records,
    build_item_banks,
    compute_feature_variance,
    construct_vectors,
    layers_label,
    load_static_construction_inputs,
    parse_layers,
    resolve_activation_root,
)
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
    parser = argparse.ArgumentParser(
        description="Evaluate static persona-vector constructions from item-level contrasts."
    )
    parser.add_argument(
        "--source-run-root",
        type=Path,
        required=True,
        help="Existing corrected-contract run root containing biography activations and bare reference rows.",
    )
    parser.add_argument(
        "--construction",
        choices=["raw_mean", "unit_mean", "diag_std_mean", "diag_var_mean"],
        default="unit_mean",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--qa-type", choices=["implicit", "explicit"], default="implicit")
    parser.add_argument("--questions-per-persona", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--remote", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--question-batch-size", type=int, default=5)
    parser.add_argument("--alphas", default="8.0")
    parser.add_argument("--layers", default="37-41")
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def chunked(items: list, chunk_size: int) -> Iterable[list]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


def default_output_dir(
    *,
    model_name: str,
    construction: str,
    layers: list[int] | None,
    questions: int,
) -> Path:
    run_id = datetime.now().strftime("%Y%m%dT%H%M%SZ")
    model_dir = model_name.replace("/", "__")
    return (
        Path("artifacts")
        / "experiments"
        / "static_vector_construction_rerun"
        / f"{run_id}__{model_dir}__{construction}__layers_{layers_label(layers)}__q{questions}"
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
    with (out_dir / "steered_only.jsonl").open("w") as handle:
        for row in steered_rows:
            handle.write(json.dumps(row.to_dict()) + "\n")
    render_summary(summary, title=f"Static construction summary - alpha={metadata['alpha']}")


def main() -> None:
    args = parse_args()
    load_dotenv()
    set_seed(args.seed)

    if args.remote and not os.environ.get("NDIF_API_KEY"):
        raise RuntimeError("NDIF_API_KEY is required for remote runs.")

    source_root = args.source_run_root
    source_metadata = json.loads((source_root / "vector_bank_metadata.json").read_text())
    model_name = args.model or source_metadata["model"]
    activation_root = resolve_activation_root(source_metadata)

    dataset = SynthPersonaDataset()
    persona_by_id = {persona.id: persona for persona in dataset}
    eval_persona_ids = source_metadata["eval_personas"]
    personas = [persona_by_id[persona_id] for persona_id in eval_persona_ids]

    sample_persona_id = eval_persona_ids[0]
    sample_records, _, _ = ActivationStore(model_name, activation_root).load_records(
        "biography", sample_persona_id
    )
    layers = parse_layers(args.layers, num_layers=sample_records.shape[1])
    steering_layers = (
        source_metadata["vectors"][sample_persona_id]["layers"]
        if layers is None
        else layers
    )

    alphas = [float(value.strip()) for value in args.alphas.split(",") if value.strip()]
    if not alphas:
        raise ValueError("Need at least one alpha")

    out_root = args.out_dir or default_output_dir(
        model_name=model_name,
        construction=args.construction,
        layers=layers,
        questions=args.questions_per_persona,
    )
    out_root.mkdir(parents=True, exist_ok=True)

    all_persona_ids, records, names = load_static_construction_inputs(
        model_name=model_name,
        activation_root=activation_root,
        layers=layers,
        center=bool(source_metadata.get("center", True)),
    )
    if missing := sorted(set(eval_persona_ids) - set(all_persona_ids)):
        raise RuntimeError(f"Eval personas missing biography activations: {missing}")

    item_banks = build_item_banks(records)
    contrasts = build_contrast_records(records=records, item_banks=item_banks)
    feature_var = compute_feature_variance(item_banks=item_banks, eps=args.eps)
    candidate_vectors = construct_vectors(
        contrasts=contrasts,
        feature_var=feature_var,
        eps=args.eps,
    )[args.construction]
    vector_bank = {
        persona_id: {
            "name": names.get(persona_id, persona_id),
            "vector": vector.unsqueeze(0),
            "layers": steering_layers,
        }
        for persona_id, vector in candidate_vectors.items()
    }

    ordered_ids = [persona.id for persona in personas]
    cross_source_for = {
        persona_id: ordered_ids[(idx + 1) % len(ordered_ids)]
        for idx, persona_id in enumerate(ordered_ids)
    }
    reference_rows = load_existing_rows(source_root / "shared_reference" / "per_example.jsonl")
    reference_rows = [
        row for row in reference_rows if row.persona_id in set(ordered_ids)
    ]
    if not reference_rows:
        raise RuntimeError(f"No bare reference rows found under {source_root}")

    (out_root / "vector_bank_metadata.json").write_text(
        json.dumps(
            {
                "source_run_root": str(source_root),
                "model": model_name,
                "construction": args.construction,
                "layers": layers if layers is not None else "all",
                "negative_variant": source_metadata.get("negative_variant"),
                "method": "static_item_contrast_construction",
                "center": source_metadata.get("center"),
                "eval_personas": ordered_ids,
                "activation_root": str(activation_root),
                "vector_pool_persona_count": len(all_persona_ids),
                "item_count": len(item_banks),
            },
            indent=2,
        )
    )

    model = StandardizedTransformer(model_name)
    for alpha in alphas:
        safe_alpha = str(alpha).replace(".", "p")
        cfg_dir = out_root / f"alpha_{safe_alpha}"
        steered_rows = load_existing_rows(cfg_dir / "steered_only.jsonl")
        failures_path = cfg_dir / "failures.json"
        failures = json.loads(failures_path.read_text()) if failures_path.exists() else []
        failed_persona_ids = {entry["persona_id"] for entry in failures}

        console.rule(f"{args.construction} layers={layers_label(layers)} alpha={alpha}")
        for persona in personas:
            if persona.id in failed_persona_ids:
                continue
            qa_pairs = select_qa_pairs(
                dataset,
                persona.id,
                args.qa_type,
                args.questions_per_persona,
            )
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
                    chunked(qa_pairs, args.question_batch_size),
                    start=1,
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
                        label=f"{args.construction} own eval for {persona.name} alpha={alpha} chunk {batch_idx}",
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
                        label=f"{args.construction} cross eval for {persona.name} alpha={alpha} chunk {batch_idx}",
                    )
                    for row in cross_batch:
                        row.condition = "steered_cross"
                    persona_rows.extend(cross_batch)
                steered_rows.extend(persona_rows)
            except Exception as exc:
                failures.append(
                    {
                        "persona_id": persona.id,
                        "persona_name": persona.name,
                        "alpha": alpha,
                        "cross_source_id": cross_id,
                        "cross_source_name": cross_payload["name"],
                        "error": str(exc),
                    }
                )
                failed_persona_ids.add(persona.id)
                console.print(f"[red]Recorded static-construction failure for {persona.name}[/]")

            write_config_outputs(
                cfg_dir,
                reference_rows=reference_rows,
                steered_rows=steered_rows,
                metadata={
                    "source_run_root": str(source_root),
                    "model": model_name,
                    "qa_type": args.qa_type,
                    "questions_per_persona": args.questions_per_persona,
                    "question_batch_size": args.question_batch_size,
                    "remote": args.remote,
                    "construction": args.construction,
                    "layers": layers if layers is not None else "all",
                    "alpha": alpha,
                    "cross_source_for": cross_source_for,
                    "negative_variant": source_metadata.get("negative_variant"),
                    "method": "static_item_contrast_construction",
                    "center": source_metadata.get("center"),
                    "vector_pool_persona_count": len(all_persona_ids),
                    "item_count": len(item_banks),
                },
                failures=failures,
            )
            console.print(f"[green]Checkpointed {args.construction} alpha={alpha} after {persona.name}[/]")


if __name__ == "__main__":
    main()
