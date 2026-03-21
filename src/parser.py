"""CLI configuration and argument parser definitions.

All *Config dataclasses and build_*_parser functions live here so that
main.py stays a thin wiring layer.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

from src.steering import STEER_LAYER

# ── Configs ──────────────────────────────────────────────────────────────────


# NOTE: This is just a possible template
@dataclass
class ExtractConfig:
    model: str
    output_dir: str


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
    activations_dir: Path
    out_dir: Path


# Parser builders ──────────────────────────────────────────────────────────


def build_extract_parser(subparsers) -> None:
    extract = subparsers.add_parser("extract", help="Extract model activations")
    extract.add_argument("--model", required=True, help="Model name or path")
    extract.add_argument("--input", required=True, help="Input data path")
    extract.add_argument("--out", required=True, help="Output directory")


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
