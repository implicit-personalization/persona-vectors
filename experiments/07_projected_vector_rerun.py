#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable

import torch
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
    parser = argparse.ArgumentParser(
        description=(
            "Reuse corrected-contract activation caches and evaluate transformed "
            "pooled-biography steering vectors without re-extracting activations."
        )
    )
    parser.add_argument(
        "--source-run-root",
        type=Path,
        required=True,
        help="Existing corrected-contract run root containing activations and bare reference rows.",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--qa-type", choices=["implicit", "explicit"], default="implicit")
    parser.add_argument("--questions-per-persona", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--remote", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--question-batch-size", type=int, default=5)
    parser.add_argument("--alphas", default="1.0,2.0")
    parser.add_argument(
        "--layers",
        default="all",
        help="Layer selection: 'all', a comma list like '25,37,38', or ranges like '37-41'.",
    )
    parser.add_argument(
        "--vector-transform",
        choices=["none", "project_mean"],
        default="project_mean",
        help="Optional vector-space transform before steering evaluation.",
    )
    parser.add_argument(
        "--center",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Override whether each saved activation vector is feature-centered before "
            "computing steering diffs. Defaults to the source run metadata."
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def parse_layers(value: str) -> list[int] | None:
    value = value.strip().lower()
    if value == "all":
        return None
    layers: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if end < start:
                raise ValueError(f"Invalid descending layer range: {part!r}")
            layers.extend(range(start, end + 1))
        else:
            layers.append(int(part))
    if not layers:
        raise ValueError("--layers selected no layers")
    return sorted(dict.fromkeys(layers))


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


def default_output_dir(
    *,
    model_name: str,
    layers_label: str,
    transform: str,
    questions: int,
) -> Path:
    run_id = datetime.now().strftime("%Y%m%dT%H%M%SZ")
    model_dir = model_name.replace("/", "__")
    return (
        Path("artifacts")
        / "experiments"
        / "projected_vector_rerun"
        / f"{run_id}__{model_dir}__{transform}__layers_{layers_label}__q{questions}"
    )


def project_out_mean(vector_bank: dict[str, dict]) -> None:
    flats = [payload["vector"].flatten().float() for payload in vector_bank.values()]
    shared = torch.stack(flats, dim=0).mean(dim=0)
    denom = shared.dot(shared) + 1e-8
    for payload in vector_bank.values():
        vector = payload["vector"].float()
        flat = vector.flatten()
        projected = flat - (flat.dot(shared) / denom) * shared
        payload["vector"] = projected.reshape_as(vector)


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
    render_summary(summary, title=f"Projected-vector summary - alpha={metadata['alpha']}")


def main() -> None:
    args = parse_args()
    load_dotenv()
    set_seed(args.seed)

    if args.remote and not os.environ.get("NDIF_API_KEY"):
        raise RuntimeError("NDIF_API_KEY is required for remote runs.")

    source_root = args.source_run_root
    source_metadata = json.loads((source_root / "vector_bank_metadata.json").read_text())
    model_name = args.model or source_metadata["model"]
    center = bool(source_metadata.get("center", True)) if args.center is None else bool(args.center)
    layers = parse_layers(args.layers)
    layer_idx = None if layers is None else layers
    layers_label = "all" if layers is None else "_".join(str(layer) for layer in layers)
    alphas = [float(value.strip()) for value in args.alphas.split(",") if value.strip()]
    if not alphas:
        raise ValueError("Need at least one alpha")

    out_root = args.out_dir or default_output_dir(
        model_name=model_name,
        layers_label=layers_label,
        transform=args.vector_transform,
        questions=args.questions_per_persona,
    )
    out_root.mkdir(parents=True, exist_ok=True)

    activation_root = Path(source_metadata["activation_root"])
    if not activation_root.is_absolute():
        activation_root = Path.cwd() / activation_root

    dataset = SynthPersonaDataset()
    persona_by_id = {persona.id: persona for persona in dataset}
    personas = [persona_by_id[persona_id] for persona_id in source_metadata["eval_personas"]]

    vector_bank: dict[str, dict] = {}
    for persona in personas:
        sv_dict = compute_steering_vector(
            persona_id=persona.id,
            model_name=model_name,
            layer_idx=layer_idx,
            activations_dir=activation_root,
            negative_variant=source_metadata.get("negative_variant", "pooled_biography"),
            method=source_metadata.get("method", "mean"),
            center=center,
            verbose=False,
        )
        vector_bank[persona.id] = {
            "name": persona.name,
            "vector": sv_dict["steering_vector"],
            "layers": sv_dict.get("layers") or layers or [sv_dict["layer"]],
            "metadata": {key: value for key, value in sv_dict.items() if key != "steering_vector"},
        }

    if args.vector_transform == "project_mean":
        project_out_mean(vector_bank)

    (out_root / "vector_bank_metadata.json").write_text(
        json.dumps(
            {
                "source_run_root": str(source_root),
                "model": model_name,
                "vector_transform": args.vector_transform,
                "layers": layers if layers is not None else "all",
                "negative_variant": source_metadata.get("negative_variant"),
                "method": source_metadata.get("method"),
                "center": center,
                "eval_personas": [persona.id for persona in personas],
                "activation_root": str(activation_root),
                "vectors": {
                    persona_id: payload["metadata"] for persona_id, payload in vector_bank.items()
                },
            },
            indent=2,
        )
    )

    ordered_ids = [persona.id for persona in personas]
    cross_source_for = {
        persona_id: ordered_ids[(idx + 1) % len(ordered_ids)]
        for idx, persona_id in enumerate(ordered_ids)
    }
    reference_rows = load_existing_rows(source_root / "shared_reference" / "per_example.jsonl")
    if not reference_rows:
        raise RuntimeError(f"No bare reference rows found under {source_root}")
    reference_rows = [
        row for row in reference_rows if row.persona_id in set(ordered_ids)
    ]

    model = StandardizedTransformer(model_name)
    for alpha in alphas:
        safe_alpha = str(alpha).replace(".", "p")
        cfg_dir = out_root / f"alpha_{safe_alpha}"
        steered_rows = load_existing_rows(cfg_dir / "steered_only.jsonl")
        failures_path = cfg_dir / "failures.json"
        failures = json.loads(failures_path.read_text()) if failures_path.exists() else []
        failed_persona_ids = {entry["persona_id"] for entry in failures}

        console.rule(f"{args.vector_transform} layers={layers_label} alpha={alpha}")
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
                        label=f"projected own eval for {persona.name} alpha={alpha} chunk {batch_idx}",
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
                        label=f"projected cross eval for {persona.name} alpha={alpha} chunk {batch_idx}",
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
                console.print(f"[red]Recorded projected-vector failure for {persona.name}[/]")

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
                    "vector_transform": args.vector_transform,
                    "layers": layers if layers is not None else "all",
                    "alpha": alpha,
                    "cross_source_for": cross_source_for,
                    "negative_variant": source_metadata.get("negative_variant"),
                    "method": source_metadata.get("method"),
                    "center": source_metadata.get("center"),
                },
                failures=failures,
            )
            console.print(f"[green]Checkpointed projected-vector alpha={alpha} after {persona.name}[/]")


if __name__ == "__main__":
    main()
