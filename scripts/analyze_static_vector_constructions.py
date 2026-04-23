#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import torch
from rich.console import Console
from rich.table import Table

from persona_vectors.artifacts import ActivationStore, list_personas
from persona_vectors.static_construction import (
    build_contrast_records,
    build_item_banks,
    compute_feature_variance,
    construct_vectors,
    cosine,
    flatten,
    layers_label,
    load_static_construction_inputs,
    parse_layers,
    resolve_activation_root,
)

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare static persona-vector constructions against item-level "
            "oracle contrast directions without making NDIF calls."
        )
    )
    parser.add_argument(
        "--source-run-root",
        type=Path,
        default=Path(
            "artifacts/experiments/cross_persona_contract_rerun/"
            "20260423T061811Z__google__gemma-2-9b-it__implicit__evalp3__vecp30__q20"
        ),
    )
    parser.add_argument(
        "--layers",
        default="37-41",
        help="Layer selection: 'all', a comma list like '25,37,38', or ranges like '37-41'.",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--eps", type=float, default=1e-5)
    return parser.parse_args()


def default_out_dir(*, model_name: str, layers: list[int] | None) -> Path:
    run_id = datetime.now().strftime("%Y%m%dT%H%M%SZ")
    model_dir = model_name.replace("/", "__")
    return (
        Path("artifacts")
        / "experiments"
        / "static_vector_construction_analysis"
        / f"{run_id}__{model_dir}__layers_{layers_label(layers)}"
    )


def pairwise_metrics(
    *,
    vectors: dict[str, torch.Tensor],
    eval_persona_ids: list[str],
    names: dict[str, str],
) -> tuple[list[dict], dict[str, dict]]:
    rows: list[dict] = []
    per_persona: dict[str, dict] = {}
    eval_vectors = [vectors[persona_id] for persona_id in eval_persona_ids]
    mean_vector = torch.stack(eval_vectors, dim=0).mean(dim=0)

    for idx, left_id in enumerate(eval_persona_ids):
        for right_id in eval_persona_ids[idx + 1 :]:
            rows.append(
                {
                    "left_persona_id": left_id,
                    "left_name": names.get(left_id, left_id),
                    "right_persona_id": right_id,
                    "right_name": names.get(right_id, right_id),
                    "cosine": cosine(vectors[left_id], vectors[right_id]),
                }
            )
    for persona_id in eval_persona_ids:
        vector = vectors[persona_id]
        per_persona[persona_id] = {
            "persona_id": persona_id,
            "persona_name": names.get(persona_id, persona_id),
            "norm": float(flatten(vector).norm().item()),
            "cosine_with_eval_mean_vector": cosine(vector, mean_vector),
        }
    return rows, per_persona


def item_alignment_metrics(
    *,
    vectors: dict[str, torch.Tensor],
    contrasts: dict[str, dict[str, torch.Tensor]],
    eval_persona_ids: list[str],
    cross_source_for: dict[str, str],
    names: dict[str, str],
) -> list[dict]:
    rows: list[dict] = []
    for persona_id in eval_persona_ids:
        own_vector = vectors[persona_id]
        cross_id = cross_source_for[persona_id]
        cross_vector = vectors[cross_id]
        item_contrasts = list(contrasts[persona_id].values())
        own_cosines = [cosine(own_vector, item_diff) for item_diff in item_contrasts]
        cross_cosines = [cosine(cross_vector, item_diff) for item_diff in item_contrasts]
        own_mean = sum(own_cosines) / len(own_cosines)
        cross_mean = sum(cross_cosines) / len(cross_cosines)
        rows.append(
            {
                "persona_id": persona_id,
                "persona_name": names.get(persona_id, persona_id),
                "cross_source_id": cross_id,
                "cross_source_name": names.get(cross_id, cross_id),
                "item_count": len(item_contrasts),
                "own_item_alignment": own_mean,
                "cross_item_alignment": cross_mean,
                "own_minus_cross_alignment": own_mean - cross_mean,
            }
        )
    return rows


def summarize_candidate(
    *,
    candidate_name: str,
    vectors: dict[str, torch.Tensor],
    contrasts: dict[str, dict[str, torch.Tensor]],
    eval_persona_ids: list[str],
    cross_source_for: dict[str, str],
    names: dict[str, str],
) -> dict:
    pairwise_rows, per_persona = pairwise_metrics(
        vectors=vectors,
        eval_persona_ids=eval_persona_ids,
        names=names,
    )
    alignment_rows = item_alignment_metrics(
        vectors=vectors,
        contrasts=contrasts,
        eval_persona_ids=eval_persona_ids,
        cross_source_for=cross_source_for,
        names=names,
    )

    pairwise_values = [row["cosine"] for row in pairwise_rows]
    mean_align_values = [
        row["cosine_with_eval_mean_vector"] for row in per_persona.values()
    ]
    own_values = [row["own_item_alignment"] for row in alignment_rows]
    cross_values = [row["cross_item_alignment"] for row in alignment_rows]
    gap_values = [row["own_minus_cross_alignment"] for row in alignment_rows]

    return {
        "candidate": candidate_name,
        "summary": {
            "mean_pairwise_cosine": sum(pairwise_values) / len(pairwise_values),
            "max_pairwise_cosine": max(pairwise_values),
            "mean_cosine_with_eval_mean_vector": sum(mean_align_values) / len(mean_align_values),
            "mean_own_item_alignment": sum(own_values) / len(own_values),
            "mean_cross_item_alignment": sum(cross_values) / len(cross_values),
            "mean_own_minus_cross_alignment": sum(gap_values) / len(gap_values),
        },
        "pairwise_cosines": pairwise_rows,
        "per_persona": list(per_persona.values()),
        "item_alignment": alignment_rows,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    source_root = args.source_run_root
    metadata = json.loads((source_root / "vector_bank_metadata.json").read_text())
    model_name = metadata["model"]
    activation_root = resolve_activation_root(metadata)

    store = ActivationStore(model_name, activation_root)
    all_persona_ids = list_personas(activation_root, model_name, ["biography"])
    eval_persona_ids = metadata["eval_personas"]
    if missing := sorted(set(eval_persona_ids) - set(all_persona_ids)):
        raise RuntimeError(f"Eval personas missing biography activations: {missing}")

    sample_vectors, _, _ = store.load_records("biography", all_persona_ids[0])
    layers = parse_layers(args.layers, num_layers=sample_vectors.shape[1])
    out_dir = args.out_dir or default_out_dir(model_name=model_name, layers=layers)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_persona_ids, records, names = load_static_construction_inputs(
        model_name=model_name,
        activation_root=activation_root,
        layers=layers,
        center=bool(metadata.get("center", True)),
    )
    item_banks = build_item_banks(records)
    contrasts = build_contrast_records(records=records, item_banks=item_banks)
    feature_var = compute_feature_variance(item_banks=item_banks, eps=args.eps)
    candidates = construct_vectors(
        contrasts=contrasts,
        feature_var=feature_var,
        eps=args.eps,
    )

    ordered_ids = list(eval_persona_ids)
    cross_source_for = {
        persona_id: ordered_ids[(idx + 1) % len(ordered_ids)]
        for idx, persona_id in enumerate(ordered_ids)
    }

    candidate_results = [
        summarize_candidate(
            candidate_name=candidate_name,
            vectors=vectors,
            contrasts=contrasts,
            eval_persona_ids=eval_persona_ids,
            cross_source_for=cross_source_for,
            names=names,
        )
        for candidate_name, vectors in candidates.items()
    ]

    summary_rows = [
        {"candidate": result["candidate"], **result["summary"]}
        for result in candidate_results
    ]
    alignment_rows = []
    pairwise_rows = []
    per_persona_rows = []
    for result in candidate_results:
        candidate = result["candidate"]
        alignment_rows.extend(
            {"candidate": candidate, **row} for row in result["item_alignment"]
        )
        pairwise_rows.extend(
            {"candidate": candidate, **row} for row in result["pairwise_cosines"]
        )
        per_persona_rows.extend(
            {"candidate": candidate, **row} for row in result["per_persona"]
        )

    output = {
        "metadata": {
            "source_run_root": str(source_root),
            "activation_root": str(activation_root),
            "model": model_name,
            "layers": layers if layers is not None else "all",
            "center": metadata.get("center"),
            "eval_personas": eval_persona_ids,
            "vector_pool_persona_count": len(all_persona_ids),
            "item_count": len(item_banks),
            "eps": args.eps,
        },
        "candidates": candidate_results,
    }
    (out_dir / "summary.json").write_text(json.dumps(output, indent=2))
    write_csv(out_dir / "candidate_summary.csv", summary_rows)
    write_csv(out_dir / "item_alignment.csv", alignment_rows)
    write_csv(out_dir / "pairwise_cosines.csv", pairwise_rows)
    write_csv(out_dir / "per_persona.csv", per_persona_rows)

    table = Table(title="Static vector construction geometry")
    table.add_column("Candidate", style="cyan")
    table.add_column("Pairwise cos", justify="right")
    table.add_column("Mean-vector cos", justify="right")
    table.add_column("Own align", justify="right")
    table.add_column("Cross align", justify="right")
    table.add_column("Own-cross gap", justify="right")
    for row in sorted(
        summary_rows,
        key=lambda item: item["mean_own_minus_cross_alignment"],
        reverse=True,
    ):
        table.add_row(
            row["candidate"],
            f"{row['mean_pairwise_cosine']:+.4f}",
            f"{row['mean_cosine_with_eval_mean_vector']:+.4f}",
            f"{row['mean_own_item_alignment']:+.4f}",
            f"{row['mean_cross_item_alignment']:+.4f}",
            f"{row['mean_own_minus_cross_alignment']:+.4f}",
        )
    console.print(table)
    console.print(f"[green]Wrote analysis to {out_dir}[/]")


if __name__ == "__main__":
    main()
