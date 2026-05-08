"""CLI entrypoint for persona-vectors.

Commands
--------
extract  Run activation extraction for all (or one) persona(s).
steer    Compute and save a steering vector from saved activations.

Examples
--------
# Extract all personas remotely (production)
python -m persona_vectors extract \\
    --model google/gemma-2-9b-it \\
    --mask-strategy answer_mean \\
    --backend remote

# Extract a single persona locally for debugging
python -m persona_vectors extract \\
    --model google/gemma-2-2b-it \\
    --persona-id <uuid>

# Compute and save steering vectors for all personas
python -m persona_vectors steer \\
    --model google/gemma-2-9b-it
"""

import sys
from dotenv import load_dotenv
from pathlib import Path

import torch
from rich.console import Console

from persona_vectors.parser import build_parser, ExtractConfig, SteerConfig

console = Console()


def _load_model(model_name: str):
    from nnterp import StandardizedTransformer

    console.print(f"Loading [bold]{model_name}[/]...")
    return StandardizedTransformer(model_name)


def run_extract(cfg: ExtractConfig) -> None:
    from persona_data.synth_persona import SynthPersonaDataset
    from persona_vectors.artifacts import ActivationStore
    from persona_vectors.extraction import run_extraction

    load_dotenv()
    torch.set_grad_enabled(False)

    remote = cfg.backend == "remote"
    model = _load_model(cfg.model)
    dataset = SynthPersonaDataset(sample_size=cfg.sample_size)
    store = ActivationStore(cfg.model)

    persona_id_set = set(cfg.persona_ids) if cfg.persona_ids else None
    personas = (
        [p for p in dataset if p.id in persona_id_set]
        if persona_id_set
        else list(dataset)
    )
    if not personas:
        console.print(f"[red]No personas found for ids {cfg.persona_ids!r}[/]")
        sys.exit(1)

    console.print(f"Extracting {len(personas)} persona(s), variants={cfg.variants}")

    for persona in personas:
        qa_pairs, _ = dataset.train_test_split(persona.id, n_train=None)

        if not cfg.force:
            todo = []
            for v in cfg.variants:
                try:
                    store.load(v, persona.id)
                    console.print(f"  [dim]skip {v} ({persona.name} already extracted)[/]")
                except FileNotFoundError:
                    todo.append(v)
            if not todo:
                continue
        else:
            todo = list(cfg.variants)

        console.rule(f"[bold]{persona.name}[/]  ({len(qa_pairs)} QA pairs)")
        try:
            results = run_extraction(
                model=model,
                model_name=cfg.model,
                persona=persona,
                qa_pairs=qa_pairs,
                variants=tuple(todo),
                mask_strategy=cfg.mask_strategy,
                remote=remote,
                verbose=cfg.verbose,
            )
            for r in results:
                console.print(
                    f"  [green]✓[/] {r.variant}: {r.n_questions} questions → {r.output_dir}"
                )
        except Exception as exc:
            console.print(f"  [red]✗ {persona.name}: {exc}[/]")

    console.print("\n[green]✓ Extraction complete[/]")


def run_steer(cfg: SteerConfig) -> None:
    from persona_data.synth_persona import SynthPersonaDataset
    from persona_vectors.artifacts import list_personas
    from persona_vectors.steering import compute_steering_vector, save_steering_vector

    load_dotenv()

    dataset = SynthPersonaDataset()
    persona_ids = (
        [cfg.persona_id]
        if cfg.persona_id
        else list_personas(cfg.activations_dir, cfg.model, ["biography", "templated"])
    )
    if not persona_ids:
        console.print("[red]No personas with saved activations found.[/]")
        sys.exit(1)

    console.print(f"Computing steering vectors for {len(persona_ids)} persona(s)")

    for persona_id in persona_ids:
        sv_dict = compute_steering_vector(
            persona_id=persona_id,
            model_name=cfg.model,
            layer_idx=cfg.layer,
            mask_strategy=cfg.mask_strategy,
            activations_dir=cfg.activations_dir,
            verbose=True,
        )
        if not sv_dict:
            console.print(f"  [yellow]Skipping {persona_id} — no activations[/]")
            continue

        out_path = Path(cfg.out_dir) / persona_id
        save_steering_vector(sv_dict, out_path)

    console.print("\n[green]✓ Steering vectors saved[/]")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "extract":
        from persona_vectors.extraction import MaskStrategy
        cfg = ExtractConfig(
            model=args.model,
            variants=args.variants,
            mask_strategy=MaskStrategy(args.mask_strategy),
            persona_ids=args.persona_id,
            sample_size=args.sample_size,
            backend=args.backend,
            verbose=args.verbose,
            force=args.force,
        )
        run_extract(cfg)

    elif args.command == "steer":
        cfg = SteerConfig(
            persona_id=getattr(args, "persona_id", None),
            model=args.model,
            layer=args.layer,
            mask_strategy=MaskStrategy(args.mask_strategy),
            activations_dir=Path(args.activations_dir),
            out_dir=Path(args.out),
        )
        run_steer(cfg)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
