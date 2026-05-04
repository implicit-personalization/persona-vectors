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
    from persona_data.prompts import BASELINE_PERSONA_ID
    from persona_data.synth_persona import SynthPersonaDataset

    from persona_vectors.extraction import run_extraction

    model = StandardizedTransformer(cfg.model)
    dataset = SynthPersonaDataset()
    personas = (
        [p for p in dataset if p.id == cfg.persona_id]
        if cfg.persona_id
        else list(dataset)
    )
    if cfg.persona_id and not personas:
        raise ValueError(f"No persona found with id {cfg.persona_id!r}")

    persona_variants = tuple(v for v in cfg.variants if v != BASELINE_PERSONA_ID)
    run_baseline = BASELINE_PERSONA_ID in cfg.variants

    common = dict(
        model=model,
        model_name=cfg.model,
        mask_strategy=cfg.mask_strategy,
        remote=cfg.remote,
        verbose=cfg.verbose,
    )

    if persona_variants:
        extracted_persona_variants = False
        for persona in tqdm(personas, desc="personas", unit="persona"):
            qa_pairs = list(dataset.get_qa(persona.id))
            if not qa_pairs:
                continue
            for r in run_extraction(
                qa_pairs=qa_pairs,
                variants=persona_variants,
                persona=persona,
                **common,
            ):
                extracted_persona_variants = True
                print(f"Saved {r.persona_name}/{r.variant} → {r.output_dir}")
        if not extracted_persona_variants:
            print(
                "No QA pairs found for selected persona(s); "
                "no persona variants extracted."
            )

    if run_baseline:
        # Baseline is persona-less; one run, sharing the first persona's QA pairs.
        baseline_qa_pairs = next(
            (qa for qa in (list(dataset.get_qa(p.id)) for p in personas) if qa),
            None,
        )
        if baseline_qa_pairs is None:
            print("Skipping baseline: no QA pairs available.")
            return
        for r in run_extraction(
            qa_pairs=baseline_qa_pairs,
            variants=(BASELINE_PERSONA_ID,),
            **common,
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
            persona_id=args.persona_id,
            remote=args.remote,
            verbose=args.verbose,
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
