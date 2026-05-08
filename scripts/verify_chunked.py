"""Verify _extract_chunked produces the same output as _extract_single_trace.

Runs both code paths on a small batch with the actual model+backend used at
extraction time, then asserts torch.allclose. Intended as a one-shot check
after touching the chunked path.

Usage:
    MODEL=google/gemma-2-9b-it BACKEND=remote python scripts/verify_chunked.py
"""

import os

import torch
from dotenv import load_dotenv
from nnterp import StandardizedTransformer

from persona_vectors.activations import _extract_chunked, _extract_single_trace


def main() -> None:
    load_dotenv()
    model_name = os.environ.get("MODEL", "google/gemma-2-9b-it")
    remote = os.environ.get("BACKEND", "remote") == "remote"

    print(f"Model={model_name} Remote={remote}")
    model = StandardizedTransformer(model_name)
    tok = model.tokenizer

    prompts = [
        "The capital of France is",
        "Photosynthesis converts light into",
    ]
    input_ids_list: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for p in prompts:
        ids = tok(p, return_tensors="pt", add_special_tokens=True).input_ids[0]
        mask = torch.ones(ids.shape[0], dtype=torch.bool)
        input_ids_list.append(ids)
        masks.append(mask)

    print("Running single-trace path...")
    single = _extract_single_trace(model, input_ids_list, masks, remote, None)

    chunk_size = max(1, model.num_layers // 4)
    print(f"Running chunked path (chunk_size={chunk_size})...")
    chunked = _extract_chunked(
        model, input_ids_list, masks, remote, None, chunk_size=chunk_size
    )

    print(f"single shape={tuple(single.shape)} chunked shape={tuple(chunked.shape)}")
    diff = (single - chunked).abs()
    print(f"max abs diff: {diff.max().item():.3e}")
    print(f"mean abs diff: {diff.mean().item():.3e}")

    if torch.allclose(single, chunked, atol=1e-3, rtol=1e-3):
        print("PASS: chunked matches single-trace within tolerance")
    else:
        print("FAIL: outputs diverge beyond tolerance")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
