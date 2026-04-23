#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import torch
from persona_data.synth_persona import SynthPersonaDataset
from rich.console import Console
from rich.table import Table

from persona_vectors.steering import compute_steering_vector

console = Console()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Compute inter-persona cosine similarity for steering vectors."
    )
    ap.add_argument("--model", default="google/gemma-2-2b-it")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument(
        "--all-layers",
        action="store_true",
        help="Use all layers and flatten the steering tensor before cosine similarity.",
    )
    ap.add_argument("--personas", type=int, default=10)
    ap.add_argument(
        "--persona-ids",
        default=None,
        help="Optional comma-separated persona ids. If set, overrides --personas ordering.",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Optional output directory. Defaults to artifacts/experiments/vector_similarity/<timestamp>.",
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
    return ap.parse_args()


def default_output_dir(model_name: str, personas: int) -> Path:
    run_id = datetime.now().strftime("%Y%m%dT%H%M%SZ")
    model_dir = model_name.replace("/", "__")
    return (
        Path("artifacts")
        / "experiments"
        / "vector_similarity"
        / f"{run_id}__{model_dir}__p{personas}"
    )


def main() -> None:
    args = parse_args()
    dataset = SynthPersonaDataset()
    if args.persona_ids:
        wanted_ids = [item.strip() for item in args.persona_ids.split(",") if item.strip()]
        persona_by_id = {persona.id: persona for persona in dataset}
        personas = [persona_by_id[persona_id] for persona_id in wanted_ids]
    else:
        personas = list(dataset)[: args.personas]
    out_dir = Path(args.out_dir) if args.out_dir else default_output_dir(args.model, args.personas)
    out_dir.mkdir(parents=True, exist_ok=True)

    vectors: list[tuple[str, str, torch.Tensor, float]] = []
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
        if not sv_dict:
            raise RuntimeError(f"Missing steering vector for {persona.id}")
        vec = sv_dict["steering_vector"].squeeze(0).detach().cpu().float().reshape(-1)
        vectors.append((persona.id, persona.name, vec, float(sv_dict["suggested_alpha"])))

    matrix_rows: list[dict[str, object]] = []
    cosine_matrix: dict[str, dict[str, float]] = {}
    for src_id, src_name, src_vec, src_alpha in vectors:
        cosine_matrix[src_id] = {}
        for tgt_id, tgt_name, tgt_vec, _ in vectors:
            cosine = float(torch.nn.functional.cosine_similarity(src_vec, tgt_vec, dim=0).item())
            cosine_matrix[src_id][tgt_id] = cosine
            matrix_rows.append(
                {
                    "source_persona_id": src_id,
                    "source_persona_name": src_name,
                    "target_persona_id": tgt_id,
                    "target_persona_name": tgt_name,
                    "cosine_similarity": cosine,
                    "source_suggested_alpha": src_alpha,
                }
            )

    with (out_dir / "pairwise_cosine.json").open("w") as f:
        json.dump(matrix_rows, f, indent=2)

    with (out_dir / "pairwise_cosine.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(matrix_rows[0].keys()))
        writer.writeheader()
        writer.writerows(matrix_rows)

    summary_rows = []
    for src_id, src_name, _, src_alpha in vectors:
        off_diag = [
            cosine
            for tgt_id, cosine in cosine_matrix[src_id].items()
            if tgt_id != src_id
        ]
        summary_rows.append(
            {
                "persona_id": src_id,
                "persona_name": src_name,
                "suggested_alpha": src_alpha,
                "mean_offdiag_cosine": sum(off_diag) / len(off_diag),
                "max_offdiag_cosine": max(off_diag),
                "min_offdiag_cosine": min(off_diag),
            }
        )

    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary_rows, f, indent=2)

    table = Table(title="Inter-persona steering-vector cosine")
    table.add_column("Persona", style="cyan")
    table.add_column("Mean offdiag cosine", style="magenta")
    table.add_column("Max offdiag cosine", style="green")
    for row in summary_rows:
        table.add_row(
            str(row["persona_name"]),
            f"{row['mean_offdiag_cosine']:.4f}",
            f"{row['max_offdiag_cosine']:.4f}",
        )
    console.print(table)
    console.print(f"[green]Saved outputs to {out_dir}[/]")


if __name__ == "__main__":
    main()
