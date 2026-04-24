#!/usr/bin/env python
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from persona_vectors.parser import (
    AnalyzeConfig,
    ExtractConfig,
    SteerConfig,
    build_parser,
)


def extract_activations(cfg: ExtractConfig) -> None:
    from nnterp import StandardizedTransformer
    from persona_data.synth_persona import SynthPersonaDataset

    from persona_vectors.extraction import run_extraction

    model = StandardizedTransformer(cfg.model)
    dataset = SynthPersonaDataset()
    personas = (
        [p for p in dataset if p.id == cfg.persona_id]
        if cfg.persona_id
        else list(dataset)
    )
    for persona in tqdm(personas, desc="personas", unit="persona"):
        qa_pairs = list(dataset.get_qa(persona.id))
        if not qa_pairs:
            continue
        results = run_extraction(
            model=model,
            model_name=cfg.model,
            persona=persona,
            qa_pairs=qa_pairs,
            variants=cfg.variants,
            mask_strategy=cfg.mask_strategy,
            remote=cfg.remote,
            verbose=cfg.verbose,
        )
        for r in results:
            print("Saved %s/%s → %s", r.persona_name, r.variant, r.output_dir)


def analyze_activations(cfg: AnalyzeConfig) -> None:
    # TODO: Load activations from disk for a specified model.
    # TODO: Compute similarity (e.g., cosine, dot).
    # TODO: Run PCA on activations and save plots/artifacts.
    raise NotImplementedError("Analysis not implemented yet")


def steer_activations(cfg: SteerConfig) -> None:
    from persona_vectors.steering import compute_steering_vector, save_steering_vector

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
    load_dotenv()

    if args.command == "extract":
        cfg = ExtractConfig(
            model=args.model,
            variants=args.variants,
            mask_strategy=args.mask_strategy,
            persona_id=args.persona_id,
            remote=args.remote,
            verbose=args.verbose,
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
