"""CLI configuration and argument parser definitions.

All *Config dataclasses and build_*_parser functions live here so that
main.py stays a thin wiring layer.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from persona_vectors.artifacts import SUPPORTED_VARIANTS
from persona_vectors.extraction import MaskStrategy
from persona_vectors.steering import STEER_LAYER

# ── Configs ──────────────────────────────────────────────────────────────────


@dataclass
class ExtractConfig:
    model: str
    variants: list[str]
    mask_strategy: MaskStrategy
    persona_id: str | None = None
    remote: bool = False
    verbose: bool = False


@dataclass
class AnalyzeConfig:
    activations_path: str
    output_dir: str
    similarity: str


@dataclass
class SteerConfig:
    persona_id: str
    model: str
    layer: int
    all_layers: bool
    negative_variant: Literal["templated", "baseline", "pooled_biography"]
    method: Literal["mean", "pca"]
    center: bool
    activations_dir: Path
    out_dir: Path


# Parser builders ──────────────────────────────────────────────────────────


def build_extract_parser(subparsers) -> None:
    extract = subparsers.add_parser("extract", help="Extract model activations")
    extract.add_argument("--model", required=True, help="HuggingFace model ID")
    extract.add_argument(
        "--variants",
        nargs="+",
        default=list(SUPPORTED_VARIANTS),
        choices=SUPPORTED_VARIANTS,
        help="Prompt variants to extract (default: all)",
    )
    extract.add_argument(
        "--mask-strategy",
        type=MaskStrategy,
        choices=list(MaskStrategy),
        default=MaskStrategy.RESPONSE_MEAN,
        help="Which tokens to average (default: response_mean)",
    )
    extract.add_argument(
        "--persona-id", default=None, help="Extract only this persona (default: all)"
    )
    extract.add_argument(
        "--remote", action="store_true", help="Execute on NDIF remote servers"
    )
    extract.add_argument(
        "--verbose", action="store_true", help="Print extraction previews"
    )


def build_analyze_parser(subparsers) -> None:
    analyze = subparsers.add_parser("analyze", help="Analyze saved activations")
    analyze.add_argument("--out", required=True, help="Output directory")
    analyze.add_argument(
        "--similarity",
        default="cosine",
        choices=["cosine", "dot"],
        help="Similarity metric",
    )


def build_steer_parser(subparsers) -> None:
    steer = subparsers.add_parser(
        "steer", help="Compute steering vector from saved activations"
    )
    steer.add_argument("--persona-id", required=True, help="Persona UUID")
    steer.add_argument("--model", default="google/gemma-2-9b-it", help="HF Model ID")
    steer.add_argument(
        "--layer", type=int, default=STEER_LAYER, help="Layer for steering vector"
    )
    steer.add_argument(
        "--all-layers",
        action="store_true",
        help="Compute one steering vector per layer and save them together.",
    )
    steer.add_argument(
        "--negative-variant",
        choices=["templated", "baseline", "pooled_biography"],
        default="baseline",
        help="Negative contrast prompt used for steering vector extraction.",
    )
    steer.add_argument(
        "--method",
        choices=["mean", "pca"],
        default="mean",
        help="How to aggregate per-question diffs into a steering direction.",
    )
    steer.add_argument(
        "--center",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Center each per-question activation vector before computing diffs.",
    )
    steer.add_argument(
        "--activations-dir",
        default="artifacts/activations",
        help="Root directory for extracted activations",
    )
    steer.add_argument(
        "--out",
        default="artifacts/vectors",
        help="Output directory for steering vectors",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract activations and analyze them (similarity + PCA)."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_extract_parser(subparsers)
    build_analyze_parser(subparsers)
    build_steer_parser(subparsers)

    return parser
