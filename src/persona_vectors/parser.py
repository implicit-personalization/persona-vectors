"""CLI configuration and argument parser definitions.

All *Config dataclasses and build_*_parser functions live here so that
main.py stays a thin wiring layer.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, get_args

from persona_vectors.artifacts import SUPPORTED_VARIANTS
from persona_vectors.extraction import MaskStrategy
from persona_vectors.steering import STEER_LAYER

Backend = Literal["local", "remote"]

# ── Configs ──────────────────────────────────────────────────────────────────


@dataclass
class ExtractConfig:
    model: str
    variants: list[str]
    mask_strategy: MaskStrategy
    persona_id: str | None = None
    backend: Backend = "local"
    verbose: bool = False


@dataclass
class AnalyzeConfig:
    model: str
    activations_dir: Path
    output_dir: Path
    variant: str
    mask_strategy: MaskStrategy
    persona_ids: list[str] | None
    layers: list[int] | None


@dataclass
class SteerConfig:
    persona_id: str
    model: str
    layer: int
    mask_strategy: MaskStrategy
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
        help=(
            "Variants to extract (default: all). 'baseline' is the shared "
            "persona-less Assistant prompt and is run once across the first "
            "selected persona's QA pairs."
        ),
    )
    extract.add_argument(
        "--mask-strategy",
        type=MaskStrategy,
        choices=list(MaskStrategy),
        default=MaskStrategy.ANSWER_MEAN,
        help="Which tokens to average (default: answer_mean)",
    )
    extract.add_argument(
        "--persona-id", default=None, help="Extract only this persona (default: all)"
    )
    extract.add_argument(
        "--backend",
        choices=get_args(Backend),
        default="local",
        help="Execution backend (default: local). 'remote' runs on NDIF.",
    )
    extract.add_argument(
        "--verbose", action="store_true", help="Print extraction previews"
    )


def build_analyze_parser(subparsers) -> None:
    analyze = subparsers.add_parser("analyze", help="Analyze saved activations")
    analyze.add_argument("--model", required=True, help="HuggingFace model ID")
    analyze.add_argument(
        "--activations-dir",
        default="artifacts/activations",
        help="Root directory containing extracted activations",
    )
    analyze.add_argument(
        "--out",
        default="artifacts/plots",
        help="Output directory for analysis plots",
    )
    analyze.add_argument(
        "--variant",
        default="biography",
        choices=SUPPORTED_VARIANTS,
        help="Artifact group to analyze (default: biography)",
    )
    analyze.add_argument(
        "--mask-strategy",
        type=MaskStrategy,
        choices=list(MaskStrategy),
        default=MaskStrategy.ANSWER_MEAN,
        help="Which saved activations to load (default: answer_mean)",
    )
    analyze.add_argument(
        "--persona-id",
        nargs="+",
        default=None,
        help="Analyze only these persona UUIDs (default: all available)",
    )
    analyze.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=None,
        help="Layers to include in interactive plots (default: all available)",
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
        "--mask-strategy",
        type=MaskStrategy,
        choices=list(MaskStrategy),
        default=MaskStrategy.ANSWER_MEAN,
        help="Which saved activations to load (default: answer_mean)",
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
