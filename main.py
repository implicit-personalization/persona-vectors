#!/usr/bin/env python
import argparse
from dataclasses import dataclass


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


def extract_activations(cfg: ExtractConfig) -> None:
    # TODO: Load model and dataset based on cfg.
    # TODO: Collect activations.
    # TODO: Save activations to disk.
    raise NotImplementedError("Extraction not implemented yet")


def analyze_activations(cfg: AnalyzeConfig) -> None:
    # TODO: Load activations from disk for a specified model.
    # TODO: Compute similarity (e.g., cosine, dot).
    # TODO: Run PCA on activations and save plots/artifacts.
    raise NotImplementedError("Analysis not implemented yet")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract activations and analyze them (similarity + PCA)."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser("extract", help="Extract model activations")
    extract.add_argument("--model", required=True, help="Model name or path")
    extract.add_argument("--input", required=True, help="Input data path")
    extract.add_argument("--out", required=True, help="Output directory")

    analyze = subparsers.add_parser("analyze", help="Analyze saved activations")
    analyze.add_argument("--out", required=True, help="Output directory")
    analyze.add_argument(
        "--similarity",
        default="cosine",
        choices=["cosine", "dot"],
        help="Similarity metric",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "extract":
        # NOTE: Load the data using the utilities in other files
        # args.input ...
        cfg = ExtractConfig(
            model=args.model,
            output_dir=args.out,
        )
        extract_activations(cfg)
    elif args.command == "analyze":
        cfg = AnalyzeConfig(
            activations_path=args.out,
            output_dir=args.out,
            similarity=args.similarity,
        )
        analyze_activations(cfg)


if __name__ == "__main__":
    main()
