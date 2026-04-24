#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

import torch
from dotenv import load_dotenv
from nnterp import StandardizedTransformer
from persona_data.environment import set_seed
from persona_data.synth_persona import PersonaData, QAPair, SynthPersonaDataset
from rich.console import Console

from persona_vectors.eval import choice_token_ids
from persona_vectors.mc_prompt_contract import render_mc_generation_prompt
from persona_vectors.static_construction import parse_layers
from persona_vectors.steering_eval_utils import run_with_remote_retry

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Causal-localization smoke test: patch clean biography final-token "
            "residual activations into corrupted bare MC prompts and measure "
            "whether gold-option logits recover."
        )
    )
    parser.add_argument("--model", default="google/gemma-2-9b-it")
    parser.add_argument(
        "--reference-jsonl",
        type=Path,
        default=Path(
            "artifacts/experiments/steering_compare/"
            "20260422T192442Z__gemma2-9b-it__p3__q20/"
            "shared_reference/per_example.jsonl"
        ),
        help="Reference rows containing bare and biography conditions used to select strong examples.",
    )
    parser.add_argument("--layers", default="30-41")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument(
        "--patch-mode",
        choices=["replace_clean", "add_clean_delta"],
        default="replace_clean",
        help=(
            "replace_clean sets the corrupted activation to the clean biography activation. "
            "add_clean_delta adds clean-biography minus corrupted-bare activation to the corrupted activation."
        ),
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--remote", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Optional .env path for NDIF_API_KEY. Useful when running from a repo without its own .env.",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def default_output_dir(
    *,
    model_name: str,
    layers_label: str,
    limit: int,
    patch_mode: str,
) -> Path:
    run_id = datetime.now().strftime("%Y%m%dT%H%M%SZ")
    model_dir = model_name.replace("/", "__")
    return (
        Path("artifacts")
        / "experiments"
        / "final_token_patch_smoke"
        / f"{run_id}__{model_dir}__{patch_mode}__layers_{layers_label}__n{limit}"
    )


def load_reference_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def select_patch_candidates(rows: list[dict], *, limit: int) -> list[dict]:
    grouped: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for row in rows:
        grouped[(row["persona_id"], row["qid"])][row["condition"]] = row

    candidates: list[dict] = []
    for (persona_id, qid), by_condition in grouped.items():
        bare = by_condition.get("bare")
        biography = by_condition.get("biography")
        if bare is None or biography is None:
            continue
        if bare["correct"] or not biography["correct"]:
            continue
        candidates.append(
            {
                "persona_id": persona_id,
                "qid": qid,
                "persona_name": bare["persona_name"],
                "question": bare["question"],
                "gold_letter": bare["gold_letter"],
                "bare_predicted_letter": bare["predicted_letter"],
                "biography_predicted_letter": biography["predicted_letter"],
                "reference_delta_gold_prob": biography["gold_prob"] - bare["gold_prob"],
            }
        )

    candidates.sort(key=lambda item: item["reference_delta_gold_prob"], reverse=True)
    return candidates[:limit]


def resolve_saved_tensor(tensor_like) -> torch.Tensor:
    if hasattr(tensor_like, "value") and getattr(tensor_like, "value") is not None:
        tensor_like = tensor_like.value
    if not isinstance(tensor_like, torch.Tensor):
        raise TypeError(f"trace result did not resolve to a tensor: {type(tensor_like)!r}")
    return tensor_like.detach().cpu()


def score_logits(
    *,
    model: StandardizedTransformer,
    prompt: str,
    prompt_len: int,
    choice_ids: list[int],
    remote: bool,
    patch_layer: int | None = None,
    patch_vector: torch.Tensor | None = None,
    patch_mode: str = "replace_clean",
) -> torch.Tensor:
    input_ids = model.tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids
    token_pos = prompt_len - 1

    with torch.no_grad(), model.trace(input_ids, remote=remote):
        if patch_layer is not None:
            if patch_vector is None:
                raise ValueError("patch_vector is required with patch_layer")
            target = model.layers_output[patch_layer][0, token_pos, :]
            if patch_mode == "replace_clean":
                target[:] = patch_vector.to(target.device)
            elif patch_mode == "add_clean_delta":
                target[:] = target + patch_vector.to(target.device)
            else:
                raise ValueError(f"Unsupported patch_mode: {patch_mode!r}")
        logits = model.logits[0, token_pos, choice_ids].float().save()

    return resolve_saved_tensor(logits)


def extract_final_activations(
    *,
    model: StandardizedTransformer,
    prompt: str,
    prompt_len: int,
    layers: list[int],
    remote: bool,
) -> torch.Tensor:
    input_ids = model.tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids
    token_pos = prompt_len - 1

    with torch.no_grad(), model.trace(input_ids, remote=remote):
        layer_vectors = []
        for layer in layers:
            layer_vectors.append(model.layers_output[layer][0, token_pos, :].float())
        stacked = torch.stack(layer_vectors, dim=0).save()

    return resolve_saved_tensor(stacked)


def logits_to_metrics(
    *,
    logits: torch.Tensor,
    choice_letters: list[str],
    gold_idx: int,
) -> dict:
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    probs = logprobs.exp()
    pred_idx = int(logprobs.argmax().item())
    other_idxs = [idx for idx in range(len(choice_letters)) if idx != gold_idx]
    best_other = logprobs[other_idxs].max().item() if other_idxs else float("-inf")
    return {
        "predicted_letter": choice_letters[pred_idx],
        "correct": pred_idx == gold_idx,
        "gold_prob": float(probs[gold_idx].item()),
        "gold_logprob": float(logprobs[gold_idx].item()),
        "margin_vs_best_other": float(logprobs[gold_idx].item() - best_other),
        "choice_probs": [float(value) for value in probs.tolist()],
        "choice_logprobs": [float(value) for value in logprobs.tolist()],
    }


def summarize(rows: list[dict]) -> dict:
    patch_rows = [row for row in rows if row["condition"] == "patched"]
    by_layer: dict[int, list[dict]] = defaultdict(list)
    for row in patch_rows:
        by_layer[int(row["patch_layer"])].append(row)

    layer_summary = {}
    for layer, layer_rows in sorted(by_layer.items()):
        n = len(layer_rows)
        layer_summary[str(layer)] = {
            "n": n,
            "accuracy": sum(1 for row in layer_rows if row["correct"]) / n if n else 0.0,
            "mean_delta_gold_prob_vs_bare": sum(
                row["delta_gold_prob_vs_bare"] for row in layer_rows
            )
            / n
            if n
            else 0.0,
            "mean_delta_margin_vs_bare": sum(
                row["delta_margin_vs_bare"] for row in layer_rows
            )
            / n
            if n
            else 0.0,
            "flip_to_gold": sum(1 for row in layer_rows if row["flip_to_gold"]),
            "mean_recovery_fraction": sum(row["gold_prob_recovery_fraction"] for row in layer_rows)
            / n
            if n
            else 0.0,
        }

    return {
        "n_patch_rows": len(patch_rows),
        "layers": layer_summary,
    }


def find_qa(dataset: SynthPersonaDataset, persona_id: str, qid: str) -> tuple[PersonaData, QAPair]:
    for persona in dataset:
        if persona.id != persona_id:
            continue
        for qa in dataset.get_qa(persona_id):
            if qa.qid == qid:
                return persona, qa
        raise KeyError(f"QID {qid!r} not found for persona {persona_id}")
    raise KeyError(f"Persona {persona_id!r} not found")


def main() -> None:
    args = parse_args()
    load_dotenv()
    if args.env_file is not None:
        load_dotenv(args.env_file, override=False)
    set_seed(args.seed)

    if args.remote and not os.environ.get("NDIF_API_KEY"):
        raise RuntimeError("NDIF_API_KEY is required for remote patching runs.")

    dataset = SynthPersonaDataset()
    reference_rows = load_reference_rows(args.reference_jsonl)
    candidates = select_patch_candidates(reference_rows, limit=args.limit)
    if not candidates:
        raise RuntimeError("No bare-wrong / biography-correct candidates found")

    model = StandardizedTransformer(args.model)
    layers = parse_layers(args.layers, num_layers=model.num_layers)
    if layers is None:
        layers = list(range(model.num_layers))
    layers_label = "_".join(str(layer) for layer in layers)

    out_dir = args.out_dir or default_output_dir(
        model_name=args.model,
        layers_label=layers_label,
        limit=args.limit,
        patch_mode=args.patch_mode,
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "model": args.model,
        "reference_jsonl": str(args.reference_jsonl),
        "layers": layers,
        "limit": args.limit,
        "remote": args.remote,
        "patch_mode": args.patch_mode,
        "patch_source": (
            "biography final prompt token"
            if args.patch_mode == "replace_clean"
            else "biography final prompt token minus bare final prompt token"
        ),
        "patch_target": "bare final prompt token",
        "candidates": candidates,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    rows: list[dict] = []
    failures: list[dict] = []

    for candidate in candidates:
        persona, qa = find_qa(dataset, candidate["persona_id"], candidate["qid"])
        if qa.correct_choice_index is None:
            raise ValueError(f"QAPair {qa.qid!r} has no correct_choice_index")
        choice_letters, choice_ids = choice_token_ids(model.tokenizer, qa)
        gold_idx = int(qa.correct_choice_index)
        bare_prompt, bare_prompt_len = render_mc_generation_prompt(
            model.tokenizer,
            persona=persona,
            qa=qa,
            condition="bare",
        )
        biography_prompt, biography_prompt_len = render_mc_generation_prompt(
            model.tokenizer,
            persona=persona,
            qa=qa,
            condition="biography",
        )

        try:
            bare_logits = run_with_remote_retry(
                lambda: score_logits(
                    model=model,
                    prompt=bare_prompt,
                    prompt_len=bare_prompt_len,
                    choice_ids=choice_ids,
                    remote=args.remote,
                ),
                label=f"bare logits {persona.name} / {qa.qid}",
            )
            biography_logits = run_with_remote_retry(
                lambda: score_logits(
                    model=model,
                    prompt=biography_prompt,
                    prompt_len=biography_prompt_len,
                    choice_ids=choice_ids,
                    remote=args.remote,
                ),
                label=f"biography logits {persona.name} / {qa.qid}",
            )
            clean_activations = run_with_remote_retry(
                lambda: extract_final_activations(
                    model=model,
                    prompt=biography_prompt,
                    prompt_len=biography_prompt_len,
                    layers=layers,
                    remote=args.remote,
                ),
                label=f"clean final activations {persona.name} / {qa.qid}",
            )
            if args.patch_mode == "add_clean_delta":
                corrupted_activations = run_with_remote_retry(
                    lambda: extract_final_activations(
                        model=model,
                        prompt=bare_prompt,
                        prompt_len=bare_prompt_len,
                        layers=layers,
                        remote=args.remote,
                    ),
                    label=f"corrupted final activations {persona.name} / {qa.qid}",
                )
                patch_activations = clean_activations - corrupted_activations
            else:
                patch_activations = clean_activations

            bare_metrics = logits_to_metrics(
                logits=bare_logits,
                choice_letters=choice_letters,
                gold_idx=gold_idx,
            )
            biography_metrics = logits_to_metrics(
                logits=biography_logits,
                choice_letters=choice_letters,
                gold_idx=gold_idx,
            )
            base_common = {
                "persona_id": persona.id,
                "persona_name": persona.name,
                "qid": qa.qid,
                "question": qa.question,
                "gold_letter": choice_letters[gold_idx],
                "choice_letters": choice_letters,
                "choices": qa.choices,
                "reference_delta_gold_prob": candidate["reference_delta_gold_prob"],
            }
            rows.append({**base_common, "condition": "bare", **bare_metrics})
            rows.append({**base_common, "condition": "biography", **biography_metrics})

            denom = biography_metrics["gold_prob"] - bare_metrics["gold_prob"]
            for layer_idx, layer in enumerate(layers):
                patch_logits = run_with_remote_retry(
                    lambda layer=layer, layer_idx=layer_idx: score_logits(
                        model=model,
                        prompt=bare_prompt,
                        prompt_len=bare_prompt_len,
                        choice_ids=choice_ids,
                        remote=args.remote,
                        patch_layer=layer,
                        patch_vector=patch_activations[layer_idx],
                        patch_mode=args.patch_mode,
                    ),
                    label=f"patch layer {layer} {persona.name} / {qa.qid}",
                )
                patch_metrics = logits_to_metrics(
                    logits=patch_logits,
                    choice_letters=choice_letters,
                    gold_idx=gold_idx,
                )
                delta_gold_prob = patch_metrics["gold_prob"] - bare_metrics["gold_prob"]
                delta_margin = (
                    patch_metrics["margin_vs_best_other"]
                    - bare_metrics["margin_vs_best_other"]
                )
                rows.append(
                    {
                        **base_common,
                        "condition": "patched",
                        "patch_layer": layer,
                        **patch_metrics,
                        "delta_gold_prob_vs_bare": delta_gold_prob,
                        "delta_margin_vs_bare": delta_margin,
                        "gold_prob_recovery_fraction": delta_gold_prob / denom
                        if abs(denom) > 1e-8
                        else 0.0,
                        "flip_to_gold": (not bare_metrics["correct"])
                        and patch_metrics["correct"],
                    }
                )
        except Exception as exc:
            failures.append(
                {
                    "persona_id": persona.id,
                    "persona_name": persona.name,
                    "qid": qa.qid,
                    "error": str(exc),
                }
            )
            console.print(f"[red]Recorded patch failure for {persona.name} / {qa.qid}[/]")

        (out_dir / "per_example.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
        (out_dir / "failures.json").write_text(json.dumps(failures, indent=2))
        summary = summarize(rows)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        console.print_json(json.dumps(summary))


if __name__ == "__main__":
    main()
