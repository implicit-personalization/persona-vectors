#!/usr/bin/env python
"""Build a Hugging Face dataset from saved activations and push it.

One row per (variant, persona). The vector is stored as a (num_layers, hidden_size) list of floats 
so consumers can `load_dataset(repo_id)` and index by persona_id without any custom loader.

Adding more personas later: re-run extraction, then re-run this script — it
rebuilds the dataset from the local artifacts and `push_to_hub` overwrites the
prior revision with the superset.

Usage:
    uv run python scripts/push_to_hf.py \
        --model google/gemma-2-9b-it \
        --repo implicit-personalization/synth-persona-vectors
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Array2D, Dataset, DatasetDict, Features, Sequence, Value
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
    rows = []
    for persona_id, entry in sorted(manifest["personas"].items()):
        tensor = load_file(str(variant_dir / f"{persona_id}.safetensors"))[
            "activations"
        ]
        rows.append(
            {
                "persona_id": persona_id,
                "name": entry["name"],
                "sample_ids": entry["sample_ids"],
                "vector": tensor.float().numpy(),
            }
        )
    return Dataset.from_list(rows, features=features)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--mask-strategy", default=DEFAULT_MASK_STRATEGY)
    parser.add_argument("--activations-dir", default="artifacts/activations")
    args = parser.parse_args()

    root = Path(args.activations_dir) / model_dir_name(args.model) / args.mask_strategy
    splits = {
        v: _build_split(root / v) for v in PERSONA_VARIANTS if (root / v).is_dir()
    }
    if not splits:
        raise SystemExit(f"no variant manifests under {root}")

    config = f"{model_dir_name(args.model)}__{args.mask_strategy}"
    DatasetDict(splits).push_to_hub(args.repo, config_name=config)
    print(
        f"pushed {config} ({list(splits)}) → https://huggingface.co/datasets/{args.repo}"
    )


if __name__ == "__main__":
    main()
