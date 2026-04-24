#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

import torch
from dotenv import load_dotenv
from nnterp import StandardizedTransformer
from persona_data.environment import set_seed
from persona_data.synth_persona import PersonaData, QAPair, SynthPersonaDataset
from rich.console import Console

from persona_vectors.mc_prompt_contract import render_mc_generation_prompt
from persona_vectors.static_construction import parse_layers, resolve_activation_root
from persona_vectors.steering import compute_steering_vector
from persona_vectors.steering_eval_utils import run_with_remote_retry, select_qa_pairs

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a minimal generation-time ActAdd smoke test. This evaluates whether "
            "previous MC-logit steering failures are caused by the apply-time contract "
            "or by the persona vector construction itself."
        )
    )
    parser.add_argument(
        "--source-run-root",
        type=Path,
        required=True,
        help="Existing corrected-contract run root containing prompt-side biography activations.",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--persona-ids",
        default=(
            "0023952f-142e-434b-82e2-7a7451b7c55f,"
            "00516d64-ab36-4367-b2c2-c992b7828861,"
            "005e1868-4e17-47e3-94fa-0d20e8d93662"
        ),
        help="Comma-separated eval persona ids. Defaults to Ethan/Gregory/Gabriela.",
    )
    parser.add_argument("--qa-type", choices=["implicit", "explicit"], default="implicit")
    parser.add_argument("--questions-per-persona", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--remote", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Optional .env path for NDIF_API_KEY. Useful when running from a repo without its own .env.",
    )
    parser.add_argument("--question-batch-size", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--alphas", default="2.0")
    parser.add_argument(
        "--layers",
        default="37-41",
        help="Layer selection for generation-time injection, e.g. '37-41' or 'all'.",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def chunked(items: list, chunk_size: int) -> Iterable[list]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


def default_output_dir(
    *,
    model_name: str,
    layers_label: str,
    questions_per_persona: int,
) -> Path:
    run_id = datetime.now().strftime("%Y%m%dT%H%M%SZ")
    model_dir = model_name.replace("/", "__")
    return (
        Path("artifacts")
        / "experiments"
        / "generation_actadd_smoke"
        / f"{run_id}__{model_dir}__layers_{layers_label}__q{questions_per_persona}"
    )


def safe_alpha_label(alpha: float) -> str:
    return str(alpha).replace(".", "p").replace("-", "m")


def resolve_saved_tensor(tensor_like) -> torch.Tensor:
    if hasattr(tensor_like, "value") and getattr(tensor_like, "value") is not None:
        tensor_like = tensor_like.value
    if not isinstance(tensor_like, torch.Tensor):
        raise TypeError(f"generation did not resolve to a tensor: {type(tensor_like)!r}")
    return tensor_like.detach().cpu()


def first_choice_letter(text: str) -> str | None:
    match = re.search(r"[A-E]", text.strip())
    return match.group(0) if match else None


def gold_letter(qa: QAPair) -> str:
    if qa.correct_choice_index is None:
        raise ValueError(f"QAPair {qa.qid!r} has no correct_choice_index")
    return chr(ord("A") + int(qa.correct_choice_index))


def normalize_steering_vector(
    steering_vector: torch.Tensor,
    layers: list[int],
) -> torch.Tensor:
    vector = steering_vector.squeeze(0).float()
    if vector.ndim == 1:
        if len(layers) != 1:
            raise ValueError("Single steering vector can only be used with one target layer")
        vector = vector.unsqueeze(0)
    if vector.ndim != 2:
        raise ValueError(f"Unsupported steering vector shape: {tuple(vector.shape)}")
    if vector.shape[0] != len(layers):
        raise ValueError(
            f"Steering vector has {vector.shape[0]} layer vectors but {len(layers)} layers were requested"
        )
    return vector


def generate_batch(
    *,
    model: StandardizedTransformer,
    prompts: list[str],
    remote: bool,
    max_new_tokens: int,
    layers: list[int] | None = None,
    steering_vector: torch.Tensor | None = None,
    alpha: float | None = None,
) -> list[str]:
    enc = model.tokenizer(
        prompts,
        return_tensors="pt",
        add_special_tokens=False,
        padding=True,
    )
    input_width = int(enc.input_ids.shape[1])

    vector = None
    if steering_vector is not None:
        if layers is None or alpha is None:
            raise ValueError("layers and alpha are required when steering_vector is provided")
        vector = normalize_steering_vector(steering_vector, layers)

    generate_kwargs = {
        "attention_mask": enc.attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "pad_token_id": model.tokenizer.pad_token_id,
        "eos_token_id": model.tokenizer.eos_token_id,
    }

    with torch.no_grad(), model.generate(
        enc.input_ids,
        remote=remote,
        **generate_kwargs,
    ) as tracer:
        if vector is not None:
            # ActAdd contract: apply to the final residual position on every
            # generation iteration, not only to the pre-generation prompt pass.
            with model.all():
                for layer_offset, layer in enumerate(layers):
                    model.layers_output[layer][:, -1, :] += alpha * vector[layer_offset].to(
                        model.layers_output[layer].device
                    )
        output = tracer.result.save()

    output_ids = resolve_saved_tensor(output)
    generations: list[str] = []
    for row in output_ids:
        generated_ids = row[input_width:]
        generations.append(
            model.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        )
    return generations


def build_generation_rows(
    *,
    persona: PersonaData,
    qa_pairs: list[QAPair],
    condition: str,
    generations: list[str],
    alpha: float | None,
    layers: list[int] | None,
    cross_source_id: str | None = None,
    cross_source_name: str | None = None,
) -> list[dict]:
    rows: list[dict] = []
    for qa, generation in zip(qa_pairs, generations, strict=True):
        prediction = first_choice_letter(generation)
        gold = gold_letter(qa)
        rows.append(
            {
                "persona_id": persona.id,
                "persona_name": persona.name,
                "qid": qa.qid,
                "question": qa.question,
                "qa_type": qa.type,
                "condition": condition,
                "gold_letter": gold,
                "predicted_letter": prediction,
                "correct": prediction == gold,
                "valid_letter": prediction is not None,
                "generation": generation,
                "choices": qa.choices,
                "alpha": alpha,
                "layers": layers,
                "cross_source_id": cross_source_id,
                "cross_source_name": cross_source_name,
            }
        )
    return rows


def summarize_rows(rows: list[dict]) -> dict:
    by_condition: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_condition[row["condition"]].append(row)

    summary: dict[str, dict] = {}
    bare_by_key = {
        (row["persona_id"], row["qid"]): row["predicted_letter"]
        for row in rows
        if row["condition"] == "bare"
    }
    bare_generation_by_key = {
        (row["persona_id"], row["qid"]): row["generation"]
        for row in rows
        if row["condition"] == "bare"
    }
    for condition, condition_rows in sorted(by_condition.items()):
        n = len(condition_rows)
        valid = sum(1 for row in condition_rows if row["valid_letter"])
        correct = sum(1 for row in condition_rows if row["correct"])
        changed_letter = sum(
            1
            for row in condition_rows
            if condition != "bare"
            and row["predicted_letter"] != bare_by_key.get((row["persona_id"], row["qid"]))
        )
        changed_generation = sum(
            1
            for row in condition_rows
            if condition != "bare"
            and row["generation"] != bare_generation_by_key.get((row["persona_id"], row["qid"]))
        )
        summary[condition] = {
            "n": n,
            "valid_letter_count": valid,
            "valid_letter_rate": valid / n if n else 0.0,
            "accuracy": correct / n if n else 0.0,
            "changed_vs_bare_rate": changed_letter / n if n and condition != "bare" else None,
            "changed_letter_vs_bare_rate": changed_letter / n if n and condition != "bare" else None,
            "changed_generation_vs_bare_rate": (
                changed_generation / n if n and condition != "bare" else None
            ),
            "letter_counts": dict(Counter(row["predicted_letter"] for row in condition_rows)),
        }
    return summary


def main() -> None:
    args = parse_args()
    load_dotenv()
    if args.env_file is not None:
        load_dotenv(args.env_file, override=False)
    set_seed(args.seed)

    if args.remote and not os.environ.get("NDIF_API_KEY"):
        raise RuntimeError("NDIF_API_KEY is required for remote generation runs.")

    source_root = args.source_run_root
    source_metadata = json.loads((source_root / "vector_bank_metadata.json").read_text())
    model_name = args.model or source_metadata["model"]
    activation_root = resolve_activation_root(source_metadata)

    dataset = SynthPersonaDataset()
    persona_by_id = {persona.id: persona for persona in dataset}
    persona_ids = [item.strip() for item in args.persona_ids.split(",") if item.strip()]
    personas = [persona_by_id[persona_id] for persona_id in persona_ids]

    sample_persona_id = persona_ids[0]
    sample_sv = compute_steering_vector(
        persona_id=sample_persona_id,
        model_name=model_name,
        layer_idx=None,
        activations_dir=activation_root,
        negative_variant=source_metadata.get("negative_variant", "pooled_biography"),
        method=source_metadata.get("method", "mean"),
        center=bool(source_metadata.get("center", True)),
        verbose=False,
    )
    layers = parse_layers(args.layers, num_layers=len(sample_sv["layers"]))
    if layers is None:
        layers = list(sample_sv["layers"])
    layers_label = "_".join(str(layer) for layer in layers)
    alphas = [float(value.strip()) for value in args.alphas.split(",") if value.strip()]
    if not alphas:
        raise ValueError("Need at least one alpha")

    out_root = args.out_dir or default_output_dir(
        model_name=model_name,
        layers_label=layers_label,
        questions_per_persona=args.questions_per_persona,
    )
    out_root.mkdir(parents=True, exist_ok=True)

    vector_bank: dict[str, dict] = {}
    for persona in personas:
        sv_dict = compute_steering_vector(
            persona_id=persona.id,
            model_name=model_name,
            layer_idx=layers,
            activations_dir=activation_root,
            negative_variant=source_metadata.get("negative_variant", "pooled_biography"),
            method=source_metadata.get("method", "mean"),
            center=bool(source_metadata.get("center", True)),
            verbose=False,
        )
        vector_bank[persona.id] = {
            "name": persona.name,
            "vector": sv_dict["steering_vector"],
            "layers": sv_dict.get("layers") or layers,
            "metadata": {key: value for key, value in sv_dict.items() if key != "steering_vector"},
        }

    ordered_ids = [persona.id for persona in personas]
    cross_source_for = {
        persona_id: ordered_ids[(idx + 1) % len(ordered_ids)]
        for idx, persona_id in enumerate(ordered_ids)
    }

    metadata = {
        "source_run_root": str(source_root),
        "model": model_name,
        "activation_root": str(activation_root),
        "qa_type": args.qa_type,
        "questions_per_persona": args.questions_per_persona,
        "question_batch_size": args.question_batch_size,
        "max_new_tokens": args.max_new_tokens,
        "remote": args.remote,
        "layers": layers,
        "alphas": alphas,
        "negative_variant": source_metadata.get("negative_variant"),
        "method": source_metadata.get("method"),
        "center": source_metadata.get("center"),
        "mask_strategy": source_metadata.get("mask_strategy"),
        "eval_personas": ordered_ids,
        "cross_source_for": cross_source_for,
        "vectors": {
            persona_id: payload["metadata"] for persona_id, payload in vector_bank.items()
        },
    }
    (out_root / "metadata.json").write_text(json.dumps(metadata, indent=2))

    model = StandardizedTransformer(model_name)

    for alpha in alphas:
        cfg_dir = out_root / f"alpha_{safe_alpha_label(alpha)}"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        rows_path = cfg_dir / "generations.jsonl"
        failures_path = cfg_dir / "failures.json"
        rows: list[dict] = []
        failures: list[dict] = []

        for persona in personas:
            qa_pairs = select_qa_pairs(
                dataset,
                persona.id,
                args.qa_type,
                args.questions_per_persona,
            )
            prompts = [
                render_mc_generation_prompt(
                    model.tokenizer,
                    persona=persona,
                    qa=qa,
                    condition="steered",
                )[0]
                for qa in qa_pairs
            ]

            try:
                for batch_idx, qa_batch in enumerate(
                    chunked(qa_pairs, args.question_batch_size),
                    start=1,
                ):
                    batch_prompts = prompts[
                        (batch_idx - 1) * args.question_batch_size : (batch_idx - 1)
                        * args.question_batch_size
                        + len(qa_batch)
                    ]
                    bare_generations = run_with_remote_retry(
                        lambda batch_prompts=batch_prompts: generate_batch(
                            model=model,
                            prompts=batch_prompts,
                            remote=args.remote,
                            max_new_tokens=args.max_new_tokens,
                        ),
                        label=f"generation bare {persona.name} chunk {batch_idx}",
                    )
                    rows.extend(
                        build_generation_rows(
                            persona=persona,
                            qa_pairs=qa_batch,
                            condition="bare",
                            generations=bare_generations,
                            alpha=None,
                            layers=None,
                        )
                    )

                    own_payload = vector_bank[persona.id]
                    own_generations = run_with_remote_retry(
                        lambda batch_prompts=batch_prompts, own_payload=own_payload: generate_batch(
                            model=model,
                            prompts=batch_prompts,
                            remote=args.remote,
                            max_new_tokens=args.max_new_tokens,
                            layers=own_payload["layers"],
                            steering_vector=own_payload["vector"],
                            alpha=alpha,
                        ),
                        label=f"generation own ActAdd {persona.name} chunk {batch_idx}",
                    )
                    rows.extend(
                        build_generation_rows(
                            persona=persona,
                            qa_pairs=qa_batch,
                            condition="steered_own",
                            generations=own_generations,
                            alpha=alpha,
                            layers=own_payload["layers"],
                        )
                    )

                    cross_id = cross_source_for[persona.id]
                    cross_payload = vector_bank[cross_id]
                    cross_generations = run_with_remote_retry(
                        lambda batch_prompts=batch_prompts, cross_payload=cross_payload: generate_batch(
                            model=model,
                            prompts=batch_prompts,
                            remote=args.remote,
                            max_new_tokens=args.max_new_tokens,
                            layers=cross_payload["layers"],
                            steering_vector=cross_payload["vector"],
                            alpha=alpha,
                        ),
                        label=f"generation cross ActAdd {persona.name} chunk {batch_idx}",
                    )
                    rows.extend(
                        build_generation_rows(
                            persona=persona,
                            qa_pairs=qa_batch,
                            condition="steered_cross",
                            generations=cross_generations,
                            alpha=alpha,
                            layers=cross_payload["layers"],
                            cross_source_id=cross_id,
                            cross_source_name=cross_payload["name"],
                        )
                    )
            except Exception as exc:
                failures.append(
                    {
                        "persona_id": persona.id,
                        "persona_name": persona.name,
                        "alpha": alpha,
                        "error": str(exc),
                    }
                )
                console.print(f"[red]Recorded generation smoke failure for {persona.name}[/]")

            rows_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            failures_path.write_text(json.dumps(failures, indent=2))
            summary = summarize_rows(rows)
            (cfg_dir / "summary.json").write_text(json.dumps(summary, indent=2))
            console.print_json(json.dumps(summary))


if __name__ == "__main__":
    main()
