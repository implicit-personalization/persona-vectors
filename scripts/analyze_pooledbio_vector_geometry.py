#!/usr/bin/env python3
"""Analyze pooled-biography steering-vector geometry from an existing run."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import torch

from persona_vectors.steering import compute_steering_vector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute pooled-biography steering vectors from cached activations "
            "and summarize pairwise cosine geometry without making NDIF calls."
        )
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="Experiment root containing vector_bank_metadata.json and activations/.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to RUN_ROOT/vector_geometry_analysis.json.",
    )
    return parser.parse_args()


def load_persona_names(run_root: Path) -> dict[str, str]:
    names: dict[str, str] = {}
    for path in [
        run_root / "alpha_1p0" / "per_example.jsonl",
        run_root / "shared_reference" / "per_example.jsonl",
    ]:
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            names.setdefault(row["persona_id"], row.get("persona_name", row["persona_id"]))
    return names


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item())


def summarize_pairwise(vectors: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for persona_a, persona_b in combinations(vectors, 2):
        va = vectors[persona_a]
        vb = vectors[persona_b]
        rows.append(
            {
                "persona_a": persona_a,
                "persona_b": persona_b,
                "all_layer_cosine": cosine(va, vb),
                "per_layer_cosine": [
                    cosine(va[:, layer, :], vb[:, layer, :])
                    for layer in range(va.shape[1])
                ],
            }
        )
    return rows


def summarize_layers(pairwise_rows: list[dict[str, Any]], vectors: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
    n_layers = next(iter(vectors.values())).shape[1]
    layer_rows: list[dict[str, Any]] = []
    for layer in range(n_layers):
        cosines = [float(row["per_layer_cosine"][layer]) for row in pairwise_rows]
        norms = [
            float(vector[:, layer, :].norm().item())
            for vector in vectors.values()
        ]
        layer_rows.append(
            {
                "layer": layer,
                "mean_pairwise_cosine": sum(cosines) / len(cosines),
                "min_pairwise_cosine": min(cosines),
                "max_pairwise_cosine": max(cosines),
                "mean_vector_norm": sum(norms) / len(norms),
            }
        )
    return layer_rows


def summarize_shared_component(vectors: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
    stacked = torch.stack([vector.flatten().float() for vector in vectors.values()], dim=0)
    shared = stacked.mean(dim=0)
    shared_norm_sq = float(shared.dot(shared).item())
    rows: list[dict[str, Any]] = []
    for persona_id, vector in vectors.items():
        flat = vector.flatten().float()
        projection = float(flat.dot(shared).item() / (shared_norm_sq + 1e-8))
        projected = projection * shared
        residual = flat - projected
        rows.append(
            {
                "persona_id": persona_id,
                "projection_coeff_on_mean_vector": projection,
                "projection_norm_fraction": float(projected.norm().item() / (flat.norm().item() + 1e-8)),
                "residual_norm_fraction": float(residual.norm().item() / (flat.norm().item() + 1e-8)),
                "cosine_with_mean_vector": cosine(flat, shared),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    run_root = args.run_root
    out_path = args.out or run_root / "vector_geometry_analysis.json"

    metadata = json.loads((run_root / "vector_bank_metadata.json").read_text())
    persona_ids = metadata["eval_personas"]
    activation_root = run_root / "activations"
    names = load_persona_names(run_root)

    vectors: dict[str, torch.Tensor] = {}
    vector_metadata: dict[str, dict[str, Any]] = {}
    for persona_id in persona_ids:
        result = compute_steering_vector(
            persona_id=persona_id,
            model_name=metadata["model"],
            layer_idx=None,
            activations_dir=activation_root,
            negative_variant="pooled_biography",
            method="mean",
            center=True,
            verbose=False,
        )
        vectors[persona_id] = result["steering_vector"].detach().cpu().float()
        vector_metadata[persona_id] = {
            key: value
            for key, value in result.items()
            if key != "steering_vector"
        }

    pairwise = summarize_pairwise(vectors)
    layer_summary = summarize_layers(pairwise, vectors)
    shared_component = summarize_shared_component(vectors)
    most_separated_layers = sorted(
        layer_summary,
        key=lambda row: (row["mean_pairwise_cosine"], -row["mean_vector_norm"]),
    )[:10]

    output = {
        "run_root": str(run_root),
        "model": metadata["model"],
        "persona_names": {persona_id: names.get(persona_id, persona_id) for persona_id in persona_ids},
        "vector_metadata": vector_metadata,
        "pairwise": pairwise,
        "layer_summary": layer_summary,
        "most_separated_layers_by_low_mean_cosine": most_separated_layers,
        "shared_component": shared_component,
    }
    out_path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"out": str(out_path), "most_separated_layers": most_separated_layers}, indent=2))


if __name__ == "__main__":
    main()
