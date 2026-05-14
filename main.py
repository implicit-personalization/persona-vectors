#!/usr/bin/env python
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from persona_vectors.parser import (
    AnalyzeConfig,
    ExtractConfig,
    ProbeConfig,
    PushConfig,
    SteerConfig,
    build_parser,
)


def extract_activations(cfg: ExtractConfig) -> None:
    from nnterp import StandardizedTransformer
    from persona_data.synth_persona import SynthPersonaDataset

    from persona_vectors.artifacts import PersonaVectorStore
    from persona_vectors.extraction import run_extraction

    dataset = SynthPersonaDataset(sample_size=cfg.sample_size)
    store = PersonaVectorStore(cfg.model, root_dir=cfg.activations_dir)
    if cfg.persona_ids is not None:
        personas = []
        for pid in cfg.persona_ids:
            match = dataset.get_persona(pid)
            if match is None:
                raise ValueError(f"No persona found with id {pid!r}")
            personas.append(match)
    else:
        personas = list(dataset)

    def select_qa_pairs(persona):
        qa_type = None if cfg.qa_type == "all" else cfg.qa_type
        if cfg.n_train is None:
            return dataset.get_qa(persona.id, type=qa_type)

        train, _ = dataset.train_test_split(persona.id, n_train=cfg.n_train)
        return train if qa_type is None else [q for q in train if q.type == qa_type]

    def matches_current_selection(persona, qa_pairs) -> bool:
        expected_sample_ids = [q.qid for q in qa_pairs]
        for variant in cfg.variants:
            stored = store.persona_sample_ids(variant, persona.id, cfg.mask_strategy)
            if stored != expected_sample_ids:
                return False
        return True

    runs = [(p, qa_pairs) for p in personas if (qa_pairs := select_qa_pairs(p))]
    if not runs:
        print("No QA pairs found for selected persona(s); nothing extracted.")
        return

    if not cfg.force:
        runs = [(p, qa) for p, qa in runs if not matches_current_selection(p, qa)]
        if not runs:
            print(
                "All requested personas already extracted for this question selection; pass --force to re-run."
            )
            return

    model = StandardizedTransformer(cfg.model)
    skipped: list[tuple[str, str]] = []
    for persona, qa_pairs in tqdm(runs, desc="personas", unit="persona"):
        try:
            results = run_extraction(
                model=model,
                model_name=cfg.model,
                qa_pairs=qa_pairs,
                variants=tuple(cfg.variants),
                persona=persona,
                mask_strategy=cfg.mask_strategy,
                remote=cfg.backend == "remote",
                verbose=cfg.verbose,
                activations_dir=cfg.activations_dir,
            )
        except Exception as e:
            if not cfg.skip_failed:
                raise
            skipped.append((persona.name, f"{type(e).__name__}: {e}"))
            print(f"Skipping {persona.name}: {type(e).__name__}: {e}")
            continue
        for r in results:
            print(f"Saved {r.persona_name}/{r.variant} → {r.output_dir}")

    if skipped:
        print(f"\nSkipped {len(skipped)} persona(s):")
        for name, reason in skipped:
            print(f"  - {name}: {reason}")


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


def push_activations(cfg: PushConfig) -> None:
    from persona_vectors.hub import push_to_hub

    try:
        push_to_hub(
            repo_id=cfg.repo,
            model_name=cfg.model,
            mask_strategy=cfg.mask_strategy,
            root_dir=cfg.activations_dir,
            variants=cfg.variants,
        )
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc


def probe_activations(cfg: ProbeConfig) -> None:
    from persona_data.synth_persona import BASELINE_PERSONA_ID, SynthPersonaDataset

    from persona_vectors.analysis import load_persona_vectors
    from persona_vectors.artifacts import PersonaVectorStore
    from persona_vectors.probes import pick_layers, run_attribute_probe

    store = PersonaVectorStore(
        cfg.model,
        root_dir=cfg.activations_dir,
        mask_strategy=cfg.mask_strategy,
    )
    persona_ids = store.list_personas([cfg.variant], mask_strategy=cfg.mask_strategy)
    if not cfg.include_baseline:
        persona_ids = [pid for pid in persona_ids if pid != BASELINE_PERSONA_ID]
    if not persona_ids:
        raise SystemExit("No personas found for the requested probe configuration.")

    samples = load_persona_vectors(store, cfg.variant, persona_ids=persona_ids)
    num_layers = int(samples.vectors.shape[1])
    layers = (
        list(range(num_layers))
        if cfg.all_layers
        else (
            cfg.layers if cfg.layers is not None else pick_layers(num_layers, fast=True)
        )
    )
    dataset = SynthPersonaDataset()
    print(
        f"Loaded {len(persona_ids)} personas; layers={num_layers}; testing layers={layers}"
    )

    for attribute in cfg.attributes:
        artifact, best, task = run_attribute_probe(
            samples,
            dataset,
            attribute,
            persona_ids,
            layers=layers,
            feature_spaces=cfg.feature_spaces,  # type: ignore[arg-type]
            n_splits=cfg.n_splits,
            min_class_count=cfg.min_class_count,
            model_name=cfg.model,
            variant=cfg.variant,
            mask_strategy=cfg.mask_strategy,
            output_dir=cfg.output_dir,
        )
        summary = (
            f"r2={best['r2']:.3f}, mae={best['mae']:.3f}"
            if task == "numeric"
            else f"balanced_accuracy={best['balanced_accuracy']:.3f}"
        )
        print(
            f"{attribute}: task={task} best={best['probe_kind']}/{best['feature_space']} "
            f"layer={best['layer']} {summary}"
        )
        print(f"  saved: {artifact.directory}")
        if artifact.pt_path is not None:
            print(f"  persona-ui .pt: {artifact.pt_path}")


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
            n_train=args.n_train,
            qa_type=args.qa_type,
            activations_dir=Path(args.activations_dir),
            backend=args.backend,
            verbose=args.verbose,
            force=args.force,
            skip_failed=args.skip_failed,
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
    elif args.command == "push":
        cfg = PushConfig(
            model=args.model,
            repo=args.repo,
            mask_strategy=args.mask_strategy,
            activations_dir=Path(args.activations_dir),
            variants=args.variants,
        )
        push_activations(cfg)
    elif args.command == "probe":
        cfg = ProbeConfig(
            model=args.model,
            activations_dir=Path(args.activations_dir),
            output_dir=Path(args.out),
            variant=args.variant,
            mask_strategy=args.mask_strategy,
            attributes=args.attributes,
            layers=args.layers,
            all_layers=args.all_layers,
            feature_spaces=args.feature_spaces,
            n_splits=args.n_splits,
            min_class_count=args.min_class_count,
            include_baseline=args.include_baseline,
        )
        probe_activations(cfg)


if __name__ == "__main__":
    main()
