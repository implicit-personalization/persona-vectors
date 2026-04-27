#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch


DEFAULT_ROOT = Path(
    "/Users/hengxuli/Repos/implicit-personalization/persona-vectors/"
    "artifacts/experiments/response_mean_direction_suite"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize response-mean attribute MC steering artifacts into a TSV. "
            "One output row is emitted per artifact and alpha."
        )
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help=(
            "Artifact directories or summary.json files. If omitted, all completed "
            "attribute-MC summaries under the default response-mean artifact root are used."
        ),
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def discover_summaries(paths: list[Path], root: Path) -> list[Path]:
    if not paths:
        paths = sorted(root.glob("*/summary.json"))
    summaries: list[Path] = []
    for path in paths:
        if path.is_dir():
            path = path / "summary.json"
        if not path.exists():
            raise FileNotFoundError(path)
        summaries.append(path)
    return summaries


def projection_metric(
    projection: dict[str, Any],
    *,
    split: str,
    vector: str,
    field: str,
) -> Any:
    key = f"split={split}::seed=1337::vector={vector}::positive_vs_negative"
    return projection.get(key, {}).get(field, "")


def mc_metric(
    mc: dict[str, Any],
    *,
    attribute: str,
    condition: str,
    alpha: float | None,
    field: str,
) -> Any:
    alpha_text = "None" if alpha is None else str(float(alpha))
    key = (
        f"attribute={attribute}::seed=1337::condition={condition}"
        f"::alpha={alpha_text}::all"
    )
    return mc.get(key, {}).get(field, "")


def available_alphas(metadata: dict[str, Any], mc: dict[str, Any]) -> list[float]:
    values = metadata.get("alphas") or []
    if values:
        return [float(value) for value in values]
    parsed: set[float] = set()
    for key in mc:
        marker = "::alpha="
        if marker not in key:
            continue
        raw = key.split(marker, 1)[1].split("::", 1)[0]
        if raw == "None":
            continue
        parsed.add(float(raw))
    return sorted(parsed)


def vector_norms(artifact_dir: Path, attribute: str) -> dict[str, Any]:
    path = artifact_dir / "vectors.pt"
    if not path.exists():
        return {
            "true_vector_norm": "",
            "shuffled_vector_norm": "",
            "shuffled_to_true_norm_ratio": "",
        }
    payload = torch.load(path, map_location="cpu")
    true_vector = payload.get(f"attribute::{attribute}::seed_1337")
    shuffled_vector = payload.get(f"attribute_control::{attribute}::seed_1337")
    true_norm = float(true_vector.float().norm().item()) if true_vector is not None else ""
    shuffled_norm = (
        float(shuffled_vector.float().norm().item()) if shuffled_vector is not None else ""
    )
    ratio: float | str = ""
    if isinstance(true_norm, float) and isinstance(shuffled_norm, float) and true_norm != 0.0:
        ratio = shuffled_norm / true_norm
    return {
        "true_vector_norm": true_norm,
        "shuffled_vector_norm": shuffled_norm,
        "shuffled_to_true_norm_ratio": ratio,
    }


def summarize_one(summary_path: Path) -> list[dict[str, Any]]:
    summary = json.loads(summary_path.read_text())
    metadata = summary.get("metadata", {})
    projection = summary.get("attribute_projection_summary", {})
    mc = summary.get("attribute_mc_summary", {})
    if not mc:
        return []

    attribute = str(metadata.get("attribute", ""))
    norms = vector_norms(summary_path.parent, attribute)
    rows: list[dict[str, Any]] = []
    for alpha in available_alphas(metadata, mc):
        rows.append(
            {
                "artifact": str(summary_path.parent),
                "layer": metadata.get("layer", ""),
                "steering_positions": metadata.get("steering_positions", "last"),
                "selected_personas": metadata.get("selected_personas", ""),
                "selected_free_response_rows": metadata.get(
                    "selected_free_response_rows", ""
                ),
                "kept_extractions": metadata.get("kept_extractions", ""),
                "skipped_extractions": metadata.get("skipped_extractions", ""),
                "attribute_mc_score_rows": metadata.get("attribute_mc_score_rows", ""),
                **norms,
                "alpha": alpha,
                "train_true_auc": projection_metric(
                    projection,
                    split="train",
                    vector="true_attribute",
                    field="pairwise_auc_positive_greater_than_negative",
                ),
                "heldout_true_auc": projection_metric(
                    projection,
                    split="heldout",
                    vector="true_attribute",
                    field="pairwise_auc_positive_greater_than_negative",
                ),
                "train_shuffled_auc": projection_metric(
                    projection,
                    split="train",
                    vector="shuffled_control",
                    field="pairwise_auc_positive_greater_than_negative",
                ),
                "heldout_shuffled_auc": projection_metric(
                    projection,
                    split="heldout",
                    vector="shuffled_control",
                    field="pairwise_auc_positive_greater_than_negative",
                ),
                "bare_margin_logprob": mc_metric(
                    mc,
                    attribute=attribute,
                    condition="bare",
                    alpha=None,
                    field="mean_positive_minus_negative_logprob",
                ),
                "true_positive_delta_logprob": mc_metric(
                    mc,
                    attribute=attribute,
                    condition="true_positive_direction",
                    alpha=alpha,
                    field="delta_vs_bare_positive_minus_negative_logprob",
                ),
                "true_negative_delta_logprob": mc_metric(
                    mc,
                    attribute=attribute,
                    condition="true_negative_direction",
                    alpha=alpha,
                    field="delta_vs_bare_positive_minus_negative_logprob",
                ),
                "shuffled_positive_delta_logprob": mc_metric(
                    mc,
                    attribute=attribute,
                    condition="shuffled_positive_direction",
                    alpha=alpha,
                    field="delta_vs_bare_positive_minus_negative_logprob",
                ),
                "shuffled_norm_matched_positive_delta_logprob": mc_metric(
                    mc,
                    attribute=attribute,
                    condition="shuffled_norm_matched_positive_direction",
                    alpha=alpha,
                    field="delta_vs_bare_positive_minus_negative_logprob",
                ),
                "true_positive_changed_letter_rate": mc_metric(
                    mc,
                    attribute=attribute,
                    condition="true_positive_direction",
                    alpha=alpha,
                    field="changed_predicted_letter_rate",
                ),
                "true_negative_changed_letter_rate": mc_metric(
                    mc,
                    attribute=attribute,
                    condition="true_negative_direction",
                    alpha=alpha,
                    field="changed_predicted_letter_rate",
                ),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    summaries = discover_summaries(args.paths, args.root)
    rows: list[dict[str, Any]] = []
    for summary_path in summaries:
        rows.extend(summarize_one(summary_path))

    fieldnames = [
        "artifact",
        "layer",
        "steering_positions",
        "selected_personas",
        "selected_free_response_rows",
        "kept_extractions",
        "skipped_extractions",
        "attribute_mc_score_rows",
        "true_vector_norm",
        "shuffled_vector_norm",
        "shuffled_to_true_norm_ratio",
        "alpha",
        "train_true_auc",
        "heldout_true_auc",
        "train_shuffled_auc",
        "heldout_shuffled_auc",
        "bare_margin_logprob",
        "true_positive_delta_logprob",
        "true_negative_delta_logprob",
        "shuffled_positive_delta_logprob",
        "shuffled_norm_matched_positive_delta_logprob",
        "true_positive_changed_letter_rate",
        "true_negative_changed_letter_rate",
    ]

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
