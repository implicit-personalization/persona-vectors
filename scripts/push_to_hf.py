#!/usr/bin/env python
"""Build a Hugging Face dataset from saved activations and push it.

Layout on the Hub:
    config_name = "{model_dir}__{mask_strategy}"   # one config per model+mask
    splits      = variant names (e.g. "templated", "biography")
    columns     = persona_id, name, sample_ids, vector (num_layers, hidden_size)

Each variant is pushed as its own split, so adding a new variant later
(e.g. biography after templated) does not clobber the existing splits in the
same config. Only the named split is rewritten. Re-running for the same
variant overwrites just that split with the current local manifest.

Usage:
    uv run python scripts/push_to_hf.py --model google/gemma-2-9b-it --repo implicit-personalization/synth-persona-vectors
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Array2D, Dataset, Features, Sequence, Value
from safetensors.torch import load_file

from persona_vectors.artifacts import (
    DEFAULT_MASK_STRATEGY,
    PERSONA_VARIANTS,
    model_dir_name,
)


def _build_split(variant_dir: Path) -> Dataset:
    manifest = json.loads((variant_dir / "manifest.json").read_text())
    num_layers, hidden = int(manifest["num_layers"]), int(manifest["hidden_size"])
    features = Features(
        {
            "persona_id": Value("string"),
            "name": Value("string"),
            "sample_ids": Sequence(Value("string")),
            "vector": Array2D(shape=(num_layers, hidden), dtype="float32"),
        }
    )
    rows = [
        {
            "persona_id": persona_id,
            "name": entry["name"],
            "sample_ids": entry["sample_ids"],
            "vector": load_file(str(variant_dir / f"{persona_id}.safetensors"))[
                "activations"
            ].float().numpy(),
        }
        for persona_id, entry in sorted(manifest["personas"].items())
    ]
    return Dataset.from_list(rows, features=features)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--mask-strategy", default=DEFAULT_MASK_STRATEGY)
    parser.add_argument("--activations-dir", default="artifacts/activations")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        choices=PERSONA_VARIANTS,
        help="Variants to push (default: all locally-available variants).",
    )
    args = parser.parse_args()

    root = Path(args.activations_dir) / model_dir_name(args.model) / args.mask_strategy
    requested = args.variants or PERSONA_VARIANTS
    variants = [v for v in requested if (root / v / "manifest.json").is_file()]
    if not variants:
        raise SystemExit(f"no variant manifests under {root}")

    config = f"{model_dir_name(args.model)}__{args.mask_strategy}"
    for variant in variants:
        _build_split(root / variant).push_to_hub(
            args.repo, config_name=config, split=variant
        )
        print(
            f"pushed {config}/{variant} "
            f"-> https://huggingface.co/datasets/{args.repo}"
        )


if __name__ == "__main__":
    main()
