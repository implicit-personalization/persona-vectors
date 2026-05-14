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
QaType = Literal["all", "explicit", "implicit"]


@dataclass
class ExtractConfig:
    model: str
    variants: list[str]
    mask_strategy: MaskStrategy
    persona_ids: list[str] | None = None
    sample_size: int | None = None
    n_train: int | None = None
    qa_type: QaType = "all"
    activations_dir: Path = Path("artifacts/activations")
    backend: Backend = "local"
    verbose: bool = False
    force: bool = False
    skip_failed: bool = False


@dataclass
class AnalyzeConfig:
    model: str
    activations_dir: Path
    output_dir: Path
    variant: str
    mask_strategy: MaskStrategy
    persona_ids: list[str] | None
    include_baseline: bool
    layers: list[int] | None


@dataclass
class PushConfig:
    model: str
    repo: str
    mask_strategy: MaskStrategy
    activations_dir: Path
    variants: list[str] | None = None


@dataclass
class SteerConfig:
    persona_id: str
    model: str
    layer: int
    mask_strategy: MaskStrategy
    activations_dir: Path
    out_dir: Path


@dataclass
class ProbeConfig:
    model: str
    activations_dir: Path
    output_dir: Path
    variant: str
    mask_strategy: MaskStrategy
    attributes: list[str]
    layers: list[int] | None
    all_layers: bool
    feature_spaces: list[str]
    n_splits: int
    min_class_count: int
    include_baseline: bool


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def build_extract_parser(subparsers) -> None:
    extract = subparsers.add_parser("extract", help="Extract model activations")
    extract.add_argument("--model", required=True, help="HuggingFace model ID")
    extract.add_argument(
        "--variants",
        nargs="+",
        default=list(SUPPORTED_VARIANTS),
        choices=SUPPORTED_VARIANTS,
        help="Prompt variant(s) to extract (default: all).",
    )
    extract.add_argument(
        "--mask-strategy",
        type=MaskStrategy,
        choices=list(MaskStrategy),
        default=MaskStrategy.ANSWER_MEAN,
        help="Which tokens to average (default: answer_mean)",
    )
    extract.add_argument(
        "--activations-dir",
        default="artifacts/activations",
        help="Root directory for extracted activations.",
    )
    selection = extract.add_mutually_exclusive_group()
    selection.add_argument(
        "--persona-id",
        nargs="+",
        default=None,
        help="Extract only these persona IDs, e.g. baseline_assistant <UUID> (default: all)",
    )
    selection.add_argument(
        "--sample-size",
        type=_positive_int,
        default=None,
        help="Load only the first N personas from the dataset (default: all)",
    )
    extract.add_argument(
        "--n-train",
        type=_positive_int,
        default=None,
        help="Use the leakage-filtered train split and cap it at N questions (default: all QA pairs).",
    )
    extract.add_argument(
        "--qa-type",
        choices=["all", "explicit", "implicit"],
        default="all",
        help="Filter QA pairs by type before extraction (default: all).",
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
    extract.add_argument(
        "--force",
        action="store_true",
        help="Re-extract personas even if already present in the local manifest.",
    )
    extract.add_argument(
        "--skip-failed",
        action="store_true",
        help="On extraction failure (e.g. remote OOM), log and continue to the next persona.",
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
        "--include-baseline",
        action="store_true",
        help="Include the persona-less Assistant baseline in discovered comparisons.",
    )
    analyze.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=None,
        help="Layers to include in interactive plots (default: all available)",
    )


def build_push_parser(subparsers) -> None:
    push = subparsers.add_parser(
        "push", help="Push saved activations to the Hugging Face Hub"
    )
    push.add_argument("--model", required=True, help="HuggingFace model ID")
    push.add_argument(
        "--repo", required=True, help="Target HF dataset repo (e.g. user/dataset)"
    )
    push.add_argument(
        "--mask-strategy",
        type=MaskStrategy,
        choices=list(MaskStrategy),
        default=MaskStrategy.ANSWER_MEAN,
        help="Which saved activations to push (default: answer_mean)",
    )
    push.add_argument(
        "--activations-dir",
        default="artifacts/activations",
        help="Root directory containing extracted activations",
    )
    push.add_argument(
        "--variants",
        nargs="+",
        default=None,
        choices=SUPPORTED_VARIANTS,
        help="Variants to push (default: all locally-available variants).",
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


def build_probe_parser(subparsers) -> None:
    probe = subparsers.add_parser(
        "probe",
        help="Train/evaluate simple linear probes from saved activations",
    )
    probe.add_argument("--model", required=True, help="HuggingFace model ID")
    probe.add_argument(
        "--activations-dir",
        default="artifacts/activations",
        help="Root directory containing extracted activations",
    )
    probe.add_argument(
        "--out",
        default="artifacts/probes",
        help="Output directory for saved probe artifacts",
    )
    probe.add_argument(
        "--variant",
        default="templated",
        choices=SUPPORTED_VARIANTS,
        help="Artifact group to probe (default: templated)",
    )
    probe.add_argument(
        "--mask-strategy",
        type=MaskStrategy,
        choices=list(MaskStrategy),
        default=MaskStrategy.ANSWER_MEAN,
        help="Which saved activations to load (default: answer_mean)",
    )
    probe.add_argument(
        "--attributes",
        nargs="+",
        default=["sex", "born_in_us", "race", "highest_degree_received", "age"],
        help="Persona attributes to probe",
    )
    probe.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=None,
        help="Specific layers to evaluate (default: five evenly-spaced layers)",
    )
    probe.add_argument(
        "--all-layers",
        action="store_true",
        help="Evaluate every available layer instead of the fast representative set.",
    )
    probe.add_argument(
        "--feature-spaces",
        nargs="+",
        choices=["raw", "pca10"],
        default=["raw"],
        help="Feature spaces to evaluate and save (default: raw).",
    )
    probe.add_argument(
        "--n-splits",
        type=_positive_int,
        default=5,
        help="Cross-validation folds (default: 5).",
    )
    probe.add_argument(
        "--min-class-count",
        type=_positive_int,
        default=5,
        help="Drop categorical/ordinal classes below this count before CV.",
    )
    probe.add_argument(
        "--include-baseline",
        action="store_true",
        help="Include baseline_assistant if present.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract activations and analyze them (similarity + PCA)."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_extract_parser(subparsers)
    build_analyze_parser(subparsers)
    build_steer_parser(subparsers)
    build_push_parser(subparsers)
    build_probe_parser(subparsers)

    return parser
