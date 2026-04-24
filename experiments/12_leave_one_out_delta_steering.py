#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
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

from persona_vectors.artifacts import ActivationStore
from persona_vectors.eval import (
    ChoiceEvalResult,
    choice_token_ids,
    score_choice_distribution,
)
from persona_vectors.mc_prompt_contract import render_mc_generation_prompt
from persona_vectors.steering import _shared_item_key
from persona_vectors.steering_eval_utils import (
    load_existing_rows,
    run_with_remote_retry,
    select_qa_pairs,
)

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Leave-one-question-out localized-delta steering diagnostic. "
            "Build per-persona vectors from biography-baseline final-token deltas "
            "excluding the held-out shared item, then evaluate own/cross steering."
        )
    )
    parser.add_argument(
        "--source-run-root",
        type=Path,
        default=Path(
            "artifacts/experiments/baseline_caa_promptlast_q20/"
            "20260424T052630Z__gemma2-9b-it__p3__q20__center_false"
        ),
        help="Run root containing vector_bank_metadata.json and activation cache.",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--layer", type=int, default=41)
    parser.add_argument("--qa-type", choices=["implicit", "explicit"], default="implicit")
    parser.add_argument("--questions-per-persona", type=int, default=20)
    parser.add_argument("--alphas", default="1.0")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--remote", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Optional .env path for NDIF_API_KEY.",
    )
    parser.add_argument(
        "--center",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override feature-centering before computing deltas. Defaults to source metadata.",
    )
    parser.add_argument(
        "--include-projected",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also evaluate leave-one-out vectors after projecting out shared mean direction.",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def default_output_dir(
    *,
    model_name: str,
    layer: int,
    questions_per_persona: int,
    center: bool,
) -> Path:
    run_id = datetime.now().strftime("%Y%m%dT%H%M%SZ")
    model_dir = model_name.replace("/", "__")
    return (
        Path("artifacts")
        / "experiments"
        / "leave_one_out_delta_steering"
        / f"{run_id}__{model_dir}__layer_{layer}__q{questions_per_persona}__center_{str(center).lower()}"
    )


def parse_alphas(value: str) -> list[float]:
    alphas = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not alphas:
        raise ValueError("Need at least one alpha")
    return alphas


def resolve_activation_root(source_metadata: dict, *, cwd: Path) -> Path:
    activation_root = Path(source_metadata["activation_root"])
    if not activation_root.is_absolute():
        activation_root = cwd / activation_root
    return activation_root


def center_features(tensor: torch.Tensor) -> torch.Tensor:
    return tensor - tensor.mean(dim=-1, keepdim=True)


def load_delta_bank(
    *,
    store: ActivationStore,
    persona_ids: list[str],
    layer: int,
    center: bool,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, str]]:
    """Load biography-baseline deltas keyed by shared item key."""

    delta_bank: dict[str, dict[str, torch.Tensor]] = {}
    persona_names: dict[str, str] = {}
    for persona_id in persona_ids:
        bio_acts, bio_qids, bio_questions = store.load_records("biography", persona_id)
        base_acts, base_qids, base_questions = store.load_records("baseline", persona_id)
        if bio_acts.shape != base_acts.shape:
            raise ValueError(f"Activation shape mismatch for {persona_id}")
        if layer < 0 or layer >= bio_acts.shape[1]:
            raise ValueError(f"Layer {layer} outside activation range for {persona_id}")
        if bio_qids != base_qids:
            raise ValueError(f"QID mismatch between biography and baseline for {persona_id}")
        if bio_questions != base_questions:
            raise ValueError(f"Question mismatch between biography and baseline for {persona_id}")

        metadata = store.load_metadata("biography", persona_id)
        persona_names[persona_id] = str(metadata.get("persona_name", persona_id))

        bio_layer = bio_acts[:, layer, :].float()
        base_layer = base_acts[:, layer, :].float()
        if center:
            bio_layer = center_features(bio_layer)
            base_layer = center_features(base_layer)
        deltas = bio_layer - base_layer

        item_map: dict[str, torch.Tensor] = {}
        for idx in range(deltas.shape[0]):
            item_key = _shared_item_key(
                qid=bio_qids[idx] if bio_qids is not None else None,
                question=bio_questions[idx],
                persona_id=persona_id,
            )
            item_map[item_key] = deltas[idx].detach().cpu()
        delta_bank[persona_id] = item_map
    return delta_bank, persona_names


def leave_one_out_vector(
    delta_bank: dict[str, dict[str, torch.Tensor]],
    *,
    persona_id: str,
    heldout_key: str,
) -> torch.Tensor:
    vectors = [
        delta
        for item_key, delta in delta_bank[persona_id].items()
        if item_key != heldout_key
    ]
    if not vectors:
        raise ValueError(f"No leave-one-out vectors for {persona_id} / {heldout_key}")
    return torch.stack(vectors, dim=0).mean(dim=0)


def project_out_shared(vectors_by_persona: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    shared = torch.stack([vector.float() for vector in vectors_by_persona.values()], dim=0).mean(
        dim=0
    )
    denom = shared.dot(shared) + 1e-8
    projected: dict[str, torch.Tensor] = {}
    for persona_id, vector in vectors_by_persona.items():
        vector = vector.float()
        projected[persona_id] = vector - (vector.dot(shared) / denom) * shared
    return projected


def metrics_from_logprobs(
    *,
    choice_letters: list[str],
    choice_logprobs: torch.Tensor,
    choice_probs: torch.Tensor,
    gold_idx: int,
) -> dict:
    predicted_idx = int(choice_logprobs.argmax().item())
    other_idxs = [idx for idx in range(len(choice_letters)) if idx != gold_idx]
    best_other = choice_logprobs[other_idxs].max().item() if other_idxs else float("-inf")
    return {
        "gold_letter": choice_letters[gold_idx],
        "predicted_letter": choice_letters[predicted_idx],
        "correct": predicted_idx == gold_idx,
        "gold_prob": float(choice_probs[gold_idx].item()),
        "gold_logprob": float(choice_logprobs[gold_idx].item()),
        "margin_vs_best_other": float(choice_logprobs[gold_idx].item() - best_other),
        "choice_letters": choice_letters,
        "choice_probs": [float(value) for value in choice_probs.tolist()],
        "choice_logprobs": [float(value) for value in choice_logprobs.tolist()],
    }


def score_steered_row(
    *,
    model: StandardizedTransformer,
    persona: PersonaData,
    qa: QAPair,
    condition: str,
    layer: int,
    vector: torch.Tensor,
    alpha: float,
    remote: bool,
    vector_source_persona_id: str,
    vector_source_persona_name: str,
    vector_source_item_key: str,
    vector_norm: float,
) -> dict:
    prompt, prompt_len = render_mc_generation_prompt(
        model.tokenizer,
        persona=persona,
        qa=qa,
        condition="steered",
    )
    choice_letters, choice_ids = choice_token_ids(model.tokenizer, qa)
    choice_logprobs, choice_probs = score_choice_distribution(
        model,
        prompt,
        prompt_len,
        choice_ids,
        remote=remote,
        steering_layer=layer,
        steering_vector=vector,
        steering_alpha=alpha,
    )
    metrics = metrics_from_logprobs(
        choice_letters=choice_letters,
        choice_logprobs=choice_logprobs,
        choice_probs=choice_probs,
        gold_idx=int(qa.correct_choice_index),
    )
    return {
        "persona_id": persona.id,
        "persona_name": persona.name,
        "qid": qa.qid,
        "item_key": _shared_item_key(qid=qa.qid, question=qa.question, persona_id=persona.id),
        "question": qa.question,
        "qa_type": qa.type,
        "condition": condition,
        "layer": layer,
        "alpha": alpha,
        "vector_source_persona_id": vector_source_persona_id,
        "vector_source_persona_name": vector_source_persona_name,
        "vector_source_item_key": vector_source_item_key,
        "vector_norm": vector_norm,
        **metrics,
    }


def bare_row_to_dict(row: ChoiceEvalResult, *, layer: int, alpha: float) -> dict:
    payload = row.to_dict()
    payload["item_key"] = _shared_item_key(
        qid=row.qid,
        question=row.question,
        persona_id=row.persona_id,
    )
    payload["layer"] = layer
    payload["alpha"] = alpha
    payload["vector_source_persona_id"] = None
    payload["vector_source_persona_name"] = None
    payload["vector_source_item_key"] = None
    payload["vector_norm"] = None
    return payload


def summarize(rows: list[dict]) -> dict:
    by_condition: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_condition[row["condition"]].append(row)

    summary: dict[str, dict] = {}
    for condition, condition_rows in sorted(by_condition.items()):
        n = len(condition_rows)
        summary[condition] = {
            "n_examples": n,
            "accuracy": sum(1 for row in condition_rows if row["correct"]) / n if n else 0.0,
            "mean_gold_prob": sum(row["gold_prob"] for row in condition_rows) / n
            if n
            else 0.0,
            "mean_gold_logprob": sum(row["gold_logprob"] for row in condition_rows) / n
            if n
            else 0.0,
            "mean_margin_vs_best_other": sum(
                row["margin_vs_best_other"] for row in condition_rows
            )
            / n
            if n
            else 0.0,
            "letter_counts": dict(Counter(row["predicted_letter"] for row in condition_rows)),
        }

    grouped: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for row in rows:
        grouped[(row["persona_id"], row["qid"])][row["condition"]] = row

    for condition in sorted(by_condition):
        if condition == "bare":
            continue
        comparable = 0
        deltas: list[float] = []
        flip_to_gold = 0
        flip_away_from_gold = 0
        any_flip = 0
        for pair in grouped.values():
            if "bare" not in pair or condition not in pair:
                continue
            bare = pair["bare"]
            steered = pair[condition]
            comparable += 1
            deltas.append(steered["gold_prob"] - bare["gold_prob"])
            if steered["predicted_letter"] != bare["predicted_letter"]:
                any_flip += 1
            if (not bare["correct"]) and steered["correct"]:
                flip_to_gold += 1
            if bare["correct"] and (not steered["correct"]):
                flip_away_from_gold += 1
        if comparable:
            summary[f"{condition}_vs_bare"] = {
                "n_examples": comparable,
                "mean_delta_gold_prob": sum(deltas) / comparable,
                "flip_to_gold": flip_to_gold,
                "flip_away_from_gold": flip_away_from_gold,
                "any_flip": any_flip,
            }

    return summary


def letter_bias(rows: list[dict]) -> dict:
    by_condition_persona: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        if row["condition"] == "bare":
            continue
        by_condition_persona[(row["condition"], row["persona_name"])].append(row)

    return {
        f"{condition}::{persona_name}": {
            "n": len(items),
            "letter_counts": dict(Counter(item["predicted_letter"] for item in items)),
            "accuracy": sum(1 for item in items if item["correct"]) / len(items)
            if items
            else 0.0,
            "mean_gold_prob": sum(item["gold_prob"] for item in items) / len(items)
            if items
            else 0.0,
        }
        for (condition, persona_name), items in sorted(by_condition_persona.items())
    }


def write_outputs(out_dir: Path, rows: list[dict], failures: list[dict], metadata: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "per_example.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "metadata": metadata,
                "summary": summarize(rows),
                "letter_bias": letter_bias(rows),
            },
            indent=2,
        )
    )
    (out_dir / "failures.json").write_text(json.dumps(failures, indent=2))


def main() -> None:
    args = parse_args()
    load_dotenv()
    if args.env_file is not None:
        load_dotenv(args.env_file, override=False)
    set_seed(args.seed)

    if args.remote and not os.environ.get("NDIF_API_KEY"):
        raise RuntimeError("NDIF_API_KEY is required for remote runs.")

    source_root = args.source_run_root
    source_metadata = json.loads((source_root / "vector_bank_metadata.json").read_text())
    model_name = args.model or source_metadata["model"]
    center = bool(source_metadata.get("center", False)) if args.center is None else bool(args.center)
    alphas = parse_alphas(args.alphas)
    activation_root = resolve_activation_root(source_metadata, cwd=Path.cwd())

    dataset = SynthPersonaDataset()
    persona_by_id = {persona.id: persona for persona in dataset}
    persona_ids = list(source_metadata["eval_personas"])
    personas = [persona_by_id[persona_id] for persona_id in persona_ids]
    store = ActivationStore(model_name, activation_root)
    delta_bank, persona_names = load_delta_bank(
        store=store,
        persona_ids=persona_ids,
        layer=args.layer,
        center=center,
    )

    out_root = args.out_dir or default_output_dir(
        model_name=model_name,
        layer=args.layer,
        questions_per_persona=args.questions_per_persona,
        center=center,
    )
    out_root.mkdir(parents=True, exist_ok=True)

    cross_source_for = {
        persona_id: persona_ids[(idx + 1) % len(persona_ids)]
        for idx, persona_id in enumerate(persona_ids)
    }

    reference_rows = load_existing_rows(source_root / "shared_reference" / "per_example.jsonl")
    bare_by_key = {
        (row.persona_id, row.qid): row
        for row in reference_rows
        if row.condition == "bare" and row.persona_id in set(persona_ids)
    }

    metadata = {
        "source_run_root": str(source_root),
        "model": model_name,
        "activation_root": str(activation_root),
        "qa_type": args.qa_type,
        "questions_per_persona": args.questions_per_persona,
        "layer": args.layer,
        "alphas": alphas,
        "center": center,
        "include_projected": args.include_projected,
        "eval_personas": persona_ids,
        "cross_source_for": cross_source_for,
        "conditions": [
            "bare",
            "same_item_delta_own",
            "loo_delta_own",
            "loo_delta_cross",
            "loo_delta_own_projected",
            "loo_delta_cross_projected",
        ]
        if args.include_projected
        else [
            "bare",
            "same_item_delta_own",
            "loo_delta_own",
            "loo_delta_cross",
        ],
    }
    (out_root / "metadata.json").write_text(json.dumps(metadata, indent=2))

    model = StandardizedTransformer(model_name)

    for alpha in alphas:
        cfg_dir = out_root / f"alpha_{str(alpha).replace('.', 'p').replace('-', 'm')}"
        rows: list[dict] = []
        failures: list[dict] = []

        for persona in personas:
            qa_pairs = select_qa_pairs(
                dataset,
                persona.id,
                args.qa_type,
                args.questions_per_persona,
            )
            for qa in qa_pairs:
                item_key = _shared_item_key(
                    qid=qa.qid,
                    question=qa.question,
                    persona_id=persona.id,
                )
                try:
                    bare_row = bare_by_key[(persona.id, qa.qid)]
                    rows.append(bare_row_to_dict(bare_row, layer=args.layer, alpha=alpha))

                    own_exact = delta_bank[persona.id][item_key]
                    rows.append(
                        run_with_remote_retry(
                            lambda own_exact=own_exact: score_steered_row(
                                model=model,
                                persona=persona,
                                qa=qa,
                                condition="same_item_delta_own",
                                layer=args.layer,
                                vector=own_exact,
                                alpha=alpha,
                                remote=args.remote,
                                vector_source_persona_id=persona.id,
                                vector_source_persona_name=persona.name,
                                vector_source_item_key=item_key,
                                vector_norm=float(own_exact.norm().item()),
                            ),
                            label=f"same-item delta {persona.name} / {qa.qid}",
                        )
                    )

                    loo_vectors = {
                        source_id: leave_one_out_vector(
                            delta_bank,
                            persona_id=source_id,
                            heldout_key=item_key,
                        )
                        for source_id in persona_ids
                    }
                    projected_vectors = (
                        project_out_shared(loo_vectors) if args.include_projected else {}
                    )

                    own_loo = loo_vectors[persona.id]
                    rows.append(
                        run_with_remote_retry(
                            lambda own_loo=own_loo: score_steered_row(
                                model=model,
                                persona=persona,
                                qa=qa,
                                condition="loo_delta_own",
                                layer=args.layer,
                                vector=own_loo,
                                alpha=alpha,
                                remote=args.remote,
                                vector_source_persona_id=persona.id,
                                vector_source_persona_name=persona.name,
                                vector_source_item_key=f"mean_except::{item_key}",
                                vector_norm=float(own_loo.norm().item()),
                            ),
                            label=f"loo own {persona.name} / {qa.qid}",
                        )
                    )

                    cross_id = cross_source_for[persona.id]
                    cross_loo = loo_vectors[cross_id]
                    rows.append(
                        run_with_remote_retry(
                            lambda cross_loo=cross_loo, cross_id=cross_id: score_steered_row(
                                model=model,
                                persona=persona,
                                qa=qa,
                                condition="loo_delta_cross",
                                layer=args.layer,
                                vector=cross_loo,
                                alpha=alpha,
                                remote=args.remote,
                                vector_source_persona_id=cross_id,
                                vector_source_persona_name=persona_names[cross_id],
                                vector_source_item_key=f"mean_except::{item_key}",
                                vector_norm=float(cross_loo.norm().item()),
                            ),
                            label=f"loo cross {persona.name} / {qa.qid}",
                        )
                    )

                    if args.include_projected:
                        own_projected = projected_vectors[persona.id]
                        rows.append(
                            run_with_remote_retry(
                                lambda own_projected=own_projected: score_steered_row(
                                    model=model,
                                    persona=persona,
                                    qa=qa,
                                    condition="loo_delta_own_projected",
                                    layer=args.layer,
                                    vector=own_projected,
                                    alpha=alpha,
                                    remote=args.remote,
                                    vector_source_persona_id=persona.id,
                                    vector_source_persona_name=persona.name,
                                    vector_source_item_key=f"projected_mean_except::{item_key}",
                                    vector_norm=float(own_projected.norm().item()),
                                ),
                                label=f"loo own projected {persona.name} / {qa.qid}",
                            )
                        )

                        cross_projected = projected_vectors[cross_id]
                        rows.append(
                            run_with_remote_retry(
                                lambda cross_projected=cross_projected, cross_id=cross_id: score_steered_row(
                                    model=model,
                                    persona=persona,
                                    qa=qa,
                                    condition="loo_delta_cross_projected",
                                    layer=args.layer,
                                    vector=cross_projected,
                                    alpha=alpha,
                                    remote=args.remote,
                                    vector_source_persona_id=cross_id,
                                    vector_source_persona_name=persona_names[cross_id],
                                    vector_source_item_key=f"projected_mean_except::{item_key}",
                                    vector_norm=float(cross_projected.norm().item()),
                                ),
                                label=f"loo cross projected {persona.name} / {qa.qid}",
                            )
                        )
                except Exception as exc:
                    failures.append(
                        {
                            "persona_id": persona.id,
                            "persona_name": persona.name,
                            "qid": qa.qid,
                            "item_key": item_key,
                            "error": str(exc),
                        }
                    )
                    console.print(f"[red]Recorded LOO steering failure for {persona.name} / {qa.qid}[/]")

                write_outputs(cfg_dir, rows, failures, {**metadata, "alpha": alpha})
            console.print(f"[green]Checkpointed alpha={alpha} after {persona.name}[/]")


if __name__ == "__main__":
    main()
