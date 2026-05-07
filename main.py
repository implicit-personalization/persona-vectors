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

    from persona_vectors.artifacts import ActivationStore
    from persona_vectors.extraction import run_extraction, select_personas_with_qa

    dataset = SynthPersonaDataset(sample_size=cfg.sample_size)
    runs = select_personas_with_qa(dataset, persona_ids=cfg.persona_ids)
    if not runs:
        print("No QA pairs found for selected persona(s); nothing extracted.")
        return

    if not cfg.force:
        done = set(
            ActivationStore(cfg.model).list_personas(
                cfg.variants, mask_strategy=cfg.mask_strategy, warn_missing=False
            )
        )
        runs = [(p, qa) for p, qa in runs if p.id not in done]
        if not runs:
            print("All requested personas already extracted; pass --force to re-run.")
            return

    model = StandardizedTransformer(cfg.model)
    for persona, qa_pairs in tqdm(runs, desc="personas", unit="persona"):
        for r in run_extraction(
            model=model,
            model_name=cfg.model,
            qa_pairs=qa_pairs,
            variants=tuple(cfg.variants),
            persona=persona,
            mask_strategy=cfg.mask_strategy,
            remote=cfg.backend == "remote",
            verbose=cfg.verbose,
        ):
            print(f"Saved {r.persona_name}/{r.variant} → {r.output_dir}")


def analyze_activations(cfg: AnalyzeConfig) -> None:
    from persona_vectors.analysis import run_saved_activation_analysis

    outputs = run_saved_activation_analysis(
        model_name=cfg.model,
        activations_dir=cfg.activations_dir,
        output_dir=cfg.output_dir,
        variant=cfg.variant,
        mask_strategy=cfg.mask_strategy,
        persona_ids=cfg.persona_ids,
        include_baseline=cfg.include_baseline,
        layers=cfg.layers,
    )
    for label, path in outputs.items():
        print(f"{label}: {path}")


def steer_activations(cfg: SteerConfig) -> None:
    from persona_vectors.steering import compute_steering_vector, save_steering_vector

    sv_dict = compute_steering_vector(
        persona_id=cfg.persona_id,
        model_name=cfg.model,
        layer_idx=cfg.layer,
        mask_strategy=cfg.mask_strategy,
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
            persona_ids=args.persona_id,
            sample_size=args.sample_size,
            backend=args.backend,
            verbose=args.verbose,
            force=args.force,
        )
        extract_activations(cfg)
    elif args.command == "analyze":
        cfg = AnalyzeConfig(
            model=args.model,
            activations_dir=Path(args.activations_dir),
            output_dir=Path(args.out),
            variant=args.variant,
            mask_strategy=args.mask_strategy,
            persona_ids=args.persona_id,
            include_baseline=args.include_baseline,
            layers=args.layers,
        )
        analyze_activations(cfg)
    elif args.command == "steer":
        cfg = SteerConfig(
            persona_id=args.persona_id,
            model=args.model,
            layer=args.layer,
            mask_strategy=args.mask_strategy,
            activations_dir=Path(args.activations_dir),
            out_dir=Path(args.out),
        )
        steer_activations(cfg)


if __name__ == "__main__":
    main()
