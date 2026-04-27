#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(
    "/Users/hengxuli/Repos/implicit-personalization/persona-vectors/"
    "artifacts/experiments/response_mean_direction_suite"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Flatten response-mean attribute MC score files into per-probe TSV rows. "
            "Use this to inspect baseline option skew and item-level steering movement."
        )
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help=(
            "Artifact directories or attribute_mc_scores.jsonl files. If omitted, "
            "all completed attribute MC score files under the default root are used."
        ),
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--conditions",
        default="",
        help="Optional comma-separated condition filter, for example 'bare,true_positive_direction'.",
    )
    return parser.parse_args()


def discover_score_files(paths: list[Path], root: Path) -> list[Path]:
    if not paths:
        paths = sorted(root.glob("*/attribute_mc_scores.jsonl"))
    score_files: list[Path] = []
    for path in paths:
        if path.is_dir():
            path = path / "attribute_mc_scores.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        score_files.append(path)
    return score_files


def load_metadata(score_path: Path) -> dict[str, Any]:
    metadata_path = score_path.parent / "metadata.json"
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text())


def load_rows(score_path: Path) -> list[dict[str, Any]]:
    with score_path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_condition_filter(raw: str) -> set[str]:
    return {item.strip() for item in raw.split(",") if item.strip()}


def flatten_score_file(
    score_path: Path,
    *,
    condition_filter: set[str],
) -> list[dict[str, Any]]:
    metadata = load_metadata(score_path)
    rows = []
    for row in load_rows(score_path):
        if condition_filter and row.get("condition") not in condition_filter:
            continue
        rows.append(
            {
                "artifact": str(score_path.parent),
                "layer": metadata.get("layer", ""),
                "steering_positions": metadata.get("steering_positions", "last"),
                "base_probe_id": row.get("base_probe_id", ""),
                "probe_id": row.get("probe_id", ""),
                "option_rotation": row.get("option_rotation", ""),
                "condition": row.get("condition", ""),
                "alpha": row.get("alpha", ""),
                "vector": row.get("vector", ""),
                "positive_letter": row.get("positive_letter", ""),
                "negative_letter": row.get("negative_letter", ""),
                "predicted_letter": row.get("predicted_letter", ""),
                "predicted_positive": row.get("predicted_positive", ""),
                "predicted_negative": row.get("predicted_negative", ""),
                "predicted_unsure": row.get("predicted_unsure", ""),
                "positive_prob": row.get("positive_prob", ""),
                "negative_prob": row.get("negative_prob", ""),
                "positive_minus_negative_prob": row.get(
                    "positive_minus_negative_prob", ""
                ),
                "positive_logprob": row.get("positive_logprob", ""),
                "negative_logprob": row.get("negative_logprob", ""),
                "positive_minus_negative_logprob": row.get(
                    "positive_minus_negative_logprob", ""
                ),
                "delta_vs_bare_positive_minus_negative_logprob": row.get(
                    "delta_vs_bare_positive_minus_negative_logprob", ""
                ),
                "delta_vs_bare_positive_minus_negative_prob": row.get(
                    "delta_vs_bare_positive_minus_negative_prob", ""
                ),
            }
        )
    rows.sort(
        key=lambda item: (
            str(item["artifact"]),
            int(item["layer"]) if item["layer"] != "" else -1,
            str(item["base_probe_id"]),
            int(item["option_rotation"]) if item["option_rotation"] != "" else -1,
            str(item["condition"]),
            float(item["alpha"]) if item["alpha"] not in ("", None) else -1.0,
        )
    )
    return rows


def main() -> None:
    args = parse_args()
    score_files = discover_score_files(args.paths, args.root)
    condition_filter = parse_condition_filter(args.conditions)
    rows: list[dict[str, Any]] = []
    for score_path in score_files:
        rows.extend(flatten_score_file(score_path, condition_filter=condition_filter))

    fieldnames = [
        "artifact",
        "layer",
        "steering_positions",
        "base_probe_id",
        "probe_id",
        "option_rotation",
        "condition",
        "alpha",
        "vector",
        "positive_letter",
        "negative_letter",
        "predicted_letter",
        "predicted_positive",
        "predicted_negative",
        "predicted_unsure",
        "positive_prob",
        "negative_prob",
        "positive_minus_negative_prob",
        "positive_logprob",
        "negative_logprob",
        "positive_minus_negative_logprob",
        "delta_vs_bare_positive_minus_negative_logprob",
        "delta_vs_bare_positive_minus_negative_prob",
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
