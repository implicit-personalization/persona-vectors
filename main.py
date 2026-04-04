#!/usr/bin/env python
from pathlib import Path

from persona_vectors.parser import (
    AnalyzeConfig,
    ExtractConfig,
    SteerConfig,
    build_parser,
)


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


def steer_activations(cfg: SteerConfig) -> None:
    from dotenv import load_dotenv

    from persona_vectors.steering import compute_steering_vector, save_steering_vector

    load_dotenv(Path(__file__).parent / ".env")

    sv_dict = compute_steering_vector(
        persona_id=cfg.persona_id,
        model_name=cfg.model,
        layer_idx=cfg.layer,
        activations_dir=cfg.activations_dir,
    )

    if not sv_dict:
        return

    out_path = cfg.out_dir / cfg.persona_id
    save_steering_vector(sv_dict, out_path)


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
    elif args.command == "steer":
        cfg = SteerConfig(
            persona_id=args.persona_id,
            model=args.model,
            layer=args.layer,
            activations_dir=Path(args.activations_dir),
            out_dir=Path(args.out),
        )
        steer_activations(cfg)


if __name__ == "__main__":
    main()
