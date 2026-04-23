#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable

import torch
from dotenv import load_dotenv
from nnterp import StandardizedTransformer
from persona_data.environment import set_seed
from persona_data.synth_persona import PersonaData, QAPair, SynthPersonaDataset
from rich.console import Console

from persona_vectors.artifacts import ActivationStore, list_personas
from persona_vectors.eval import ChoiceEvalResult, choice_token_ids
from persona_vectors.mc_prompt_contract import render_mc_generation_prompt
from persona_vectors.steering import _shared_item_key
from persona_vectors.steering_eval_utils import (
    load_existing_rows,
    render_summary,
    run_with_remote_retry,
    select_qa_pairs,
    summarize,
    write_outputs,
)

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate an oracle item-conditioned contrast vector. This reuses "
            "existing biography activation caches and applies each shared MC "
            "item's own contrast vector at answer time."
        )
    )
    parser.add_argument(
        "--source-run-root",
        type=Path,
        required=True,
        help="Existing corrected-contract run root containing biography activations and bare rows.",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--qa-type", choices=["implicit", "explicit"], default="implicit")
    parser.add_argument("--questions-per-persona", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--remote", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--question-batch-size", type=int, default=5)
    parser.add_argument("--alphas", default="8.0")
    parser.add_argument(
        "--layers",
        default="37-41",
        help="Layer selection: 'all', a comma list like '25,37,38', or ranges like '37-41'.",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def parse_layers(value: str) -> list[int] | None:
    value = value.strip().lower()
    if value == "all":
        return None
    layers: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if end < start:
                raise ValueError(f"Invalid descending layer range: {part!r}")
            layers.extend(range(start, end + 1))
        else:
            layers.append(int(part))
    if not layers:
        raise ValueError("--layers selected no layers")
    return sorted(dict.fromkeys(layers))


def layers_label(layers: list[int] | None) -> str:
    return "all" if layers is None else "_".join(str(layer) for layer in layers)


def chunked(items: list, chunk_size: int) -> Iterable[list]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


def default_output_dir(
    *,
    model_name: str,
    layers: list[int] | None,
    questions: int,
) -> Path:
    run_id = datetime.now().strftime("%Y%m%dT%H%M%SZ")
    model_dir = model_name.replace("/", "__")
    return (
        Path("artifacts")
        / "experiments"
        / "item_conditioned_oracle"
        / f"{run_id}__{model_dir}__layers_{layers_label(layers)}__q{questions}"
    )


def normalize_features(tensor: torch.Tensor) -> torch.Tensor:
    return tensor - tensor.mean(dim=-1, keepdim=True)


def build_item_vector_bank(
    *,
    store: ActivationStore,
    persona_ids: list[str],
    layers: list[int] | None,
    center: bool,
) -> dict[str, dict[str, torch.Tensor]]:
    """Build per-persona, per-shared-item contrast vectors.

    The vector for source persona P and item I is:
      biography_activation(P, I) - mean biography_activation(other personas, I)

    Unlike pooled_biography, this does not average across items. It is an
    oracle check for whether item-level persona signal exists in the cache.
    """

    records: dict[str, dict[str, torch.Tensor]] = {}
    for persona_id in persona_ids:
        activations, qids, questions = store.load_records("biography", persona_id)
        selected = activations.float() if layers is None else activations[:, layers, :].float()
        if center:
            selected = normalize_features(selected)
        item_map: dict[str, torch.Tensor] = {}
        for idx in range(selected.shape[0]):
            item_key = _shared_item_key(
                qid=qids[idx] if qids is not None else None,
                question=questions[idx],
                persona_id=persona_id,
            )
            item_map[item_key] = selected[idx]
        records[persona_id] = item_map

    item_negative_bank: dict[str, list[tuple[str, torch.Tensor]]] = {}
    for persona_id, item_map in records.items():
        for item_key, activation in item_map.items():
            item_negative_bank.setdefault(item_key, []).append((persona_id, activation))

    vector_bank: dict[str, dict[str, torch.Tensor]] = {}
    for persona_id, item_map in records.items():
        source_vectors: dict[str, torch.Tensor] = {}
        for item_key, pos in item_map.items():
            negatives = [
                activation
                for other_id, activation in item_negative_bank.get(item_key, [])
                if other_id != persona_id
            ]
            if not negatives:
                continue
            source_vectors[item_key] = pos - torch.stack(negatives, dim=0).mean(dim=0)
        vector_bank[persona_id] = source_vectors
    return vector_bank


def score_item_conditioned_batch(
    model: StandardizedTransformer,
    persona: PersonaData,
    qa_pairs: list[QAPair],
    *,
    condition_name: str,
    steering_vectors: torch.Tensor,
    steering_layers: list[int],
    steering_alpha: float,
    remote: bool,
) -> list[ChoiceEvalResult]:
    prompts: list[str] = []
    prompt_lens: list[int] = []
    choice_letters_list: list[list[str]] = []
    choice_ids_list: list[list[int]] = []
    for qa in qa_pairs:
        if qa.answer_format != "choice" or qa.correct_choice_index is None:
            raise ValueError(f"QAPair {qa.qid!r} is not a scored multiple-choice item")
        prompt, prompt_len = render_mc_generation_prompt(
            model.tokenizer,
            persona=persona,
            qa=qa,
            condition="steered",
        )
        choice_letters, token_ids = choice_token_ids(model.tokenizer, qa)
        prompts.append(prompt)
        prompt_lens.append(prompt_len)
        choice_letters_list.append(choice_letters)
        choice_ids_list.append(token_ids)

    if steering_vectors.ndim != 3:
        raise ValueError(
            f"Expected steering_vectors shape (batch, layers, hidden), got {tuple(steering_vectors.shape)}"
        )
    if steering_vectors.shape[0] != len(prompts):
        raise ValueError("steering vector batch size does not match prompts")
    if steering_vectors.shape[1] != len(steering_layers):
        raise ValueError("steering vector layer count does not match steering_layers")

    enc = model.tokenizer(
        prompts,
        return_tensors="pt",
        add_special_tokens=False,
        padding=True,
    )
    input_ids = enc.input_ids
    attention_mask = enc.attention_mask
    token_positions = [prompt_len - 1 for prompt_len in prompt_lens]
    batch_index = list(range(len(prompts)))

    with torch.no_grad(), model.trace(input_ids, attention_mask=attention_mask, remote=remote):
        for layer_offset, layer in enumerate(steering_layers):
            model.steer(
                layers=layer,
                steering_vector=steering_vectors[:, layer_offset, :],
                factor=steering_alpha,
                token_positions=token_positions,
                batch_index=batch_index,
            )

        batch_logprobs = []
        for batch_idx, (token_position, choice_ids) in enumerate(
            zip(token_positions, choice_ids_list, strict=True)
        ):
            choice_logits = model.logits[batch_idx, token_position, choice_ids].float()
            batch_logprobs.append(torch.log_softmax(choice_logits, dim=-1))
        saved_logprobs = torch.stack(batch_logprobs, dim=0).save()

    resolved_tensor = (
        saved_logprobs.value
        if hasattr(saved_logprobs, "value") and saved_logprobs.value is not None
        else saved_logprobs
    )
    if not isinstance(resolved_tensor, torch.Tensor):
        raise TypeError(
            f"choice scoring did not resolve to a tensor: {type(resolved_tensor)!r}"
        )
    resolved = resolved_tensor.detach().cpu()
    rows: list[ChoiceEvalResult] = []
    for idx, (qa, choice_letters) in enumerate(
        zip(qa_pairs, choice_letters_list, strict=True)
    ):
        choice_logprobs = resolved[idx]
        choice_probs = choice_logprobs.exp()
        gold_idx = qa.correct_choice_index
        assert gold_idx is not None
        top_idx = int(choice_logprobs.argmax().item())
        other_idxs = [i for i in range(len(choice_letters)) if i != gold_idx]
        best_other = (
            choice_logprobs[other_idxs].max().item() if other_idxs else float("-inf")
        )
        rows.append(
            ChoiceEvalResult(
                persona_id=persona.id,
                persona_name=persona.name,
                qid=qa.qid,
                question=qa.question,
                qa_type=qa.type,
                condition=condition_name,
                gold_letter=choice_letters[gold_idx],
                predicted_letter=choice_letters[top_idx],
                correct=top_idx == gold_idx,
                gold_prob=float(choice_probs[gold_idx].item()),
                gold_logprob=float(choice_logprobs[gold_idx].item()),
                margin_vs_best_other=float(choice_logprobs[gold_idx].item() - best_other),
                choice_letters=choice_letters,
                choice_probs=[float(v) for v in choice_probs.tolist()],
                choice_logprobs=[float(v) for v in choice_logprobs.tolist()],
                steering_layer=steering_layers,
                steering_alpha=steering_alpha,
            )
        )
    return rows


def item_key_for_qa(qa: QAPair, persona_id: str) -> str:
    return _shared_item_key(qid=qa.qid, question=qa.question, persona_id=persona_id)


def write_config_outputs(
    out_dir: Path,
    *,
    reference_rows: list[ChoiceEvalResult],
    steered_rows: list[ChoiceEvalResult],
    metadata: dict,
    failures: list[dict],
) -> None:
    merged_rows = reference_rows + steered_rows
    summary = summarize(merged_rows)
    write_outputs(
        out_dir,
        merged_rows,
        summary,
        metadata,
        failures,
        jsonl_name="per_example.jsonl",
        csv_name="per_example.csv",
    )
    with (out_dir / "steered_only.jsonl").open("w") as handle:
        for row in steered_rows:
            handle.write(json.dumps(row.to_dict()) + "\n")
    render_summary(summary, title=f"Item-conditioned oracle summary - alpha={metadata['alpha']}")


def main() -> None:
    args = parse_args()
    load_dotenv()
    set_seed(args.seed)

    if args.remote and not os.environ.get("NDIF_API_KEY"):
        raise RuntimeError("NDIF_API_KEY is required for remote runs.")

    source_root = args.source_run_root
    source_metadata = json.loads((source_root / "vector_bank_metadata.json").read_text())
    model_name = args.model or source_metadata["model"]
    selected_layers = parse_layers(args.layers)
    steering_layers = (
        source_metadata["vectors"][source_metadata["eval_personas"][0]]["layers"]
        if selected_layers is None
        else selected_layers
    )
    alphas = [float(value.strip()) for value in args.alphas.split(",") if value.strip()]
    if not alphas:
        raise ValueError("Need at least one alpha")

    out_root = args.out_dir or default_output_dir(
        model_name=model_name,
        layers=selected_layers,
        questions=args.questions_per_persona,
    )
    out_root.mkdir(parents=True, exist_ok=True)

    activation_root = Path(source_metadata["activation_root"])
    if not activation_root.is_absolute():
        activation_root = Path.cwd() / activation_root

    store = ActivationStore(model_name, activation_root)
    vector_pool_personas = list_personas(activation_root, model_name, ["biography"])
    eval_persona_ids = source_metadata["eval_personas"]
    if missing := sorted(set(eval_persona_ids) - set(vector_pool_personas)):
        raise RuntimeError(f"Eval personas missing biography activations: {missing}")

    console.print(
        f"[cyan]Building item-conditioned vector bank from {len(vector_pool_personas)} biography personas[/]"
    )
    vector_bank = build_item_vector_bank(
        store=store,
        persona_ids=vector_pool_personas,
        layers=selected_layers,
        center=bool(source_metadata.get("center", True)),
    )

    dataset = SynthPersonaDataset()
    persona_by_id = {persona.id: persona for persona in dataset}
    personas = [persona_by_id[persona_id] for persona_id in eval_persona_ids]
    ordered_ids = [persona.id for persona in personas]
    cross_source_for = {
        persona_id: ordered_ids[(idx + 1) % len(ordered_ids)]
        for idx, persona_id in enumerate(ordered_ids)
    }

    qa_by_persona = {
        persona.id: select_qa_pairs(
            dataset,
            persona.id,
            args.qa_type,
            args.questions_per_persona,
        )
        for persona in personas
    }
    wanted_pairs = {
        (persona_id, qa.qid)
        for persona_id, qa_pairs in qa_by_persona.items()
        for qa in qa_pairs
    }
    reference_rows = [
        row
        for row in load_existing_rows(source_root / "shared_reference" / "per_example.jsonl")
        if (row.persona_id, row.qid) in wanted_pairs
    ]
    if len(reference_rows) != len(wanted_pairs):
        raise RuntimeError(
            f"Bare reference rows incomplete: got {len(reference_rows)} expected {len(wanted_pairs)}"
        )

    (out_root / "vector_bank_metadata.json").write_text(
        json.dumps(
            {
                "source_run_root": str(source_root),
                "model": model_name,
                "experiment": "item_conditioned_oracle",
                "layers": selected_layers if selected_layers is not None else "all",
                "negative_variant": source_metadata.get("negative_variant"),
                "method": "per_item_mean_contrast",
                "center": source_metadata.get("center"),
                "vector_pool_persona_count": len(vector_pool_personas),
                "eval_personas": ordered_ids,
                "activation_root": str(activation_root),
            },
            indent=2,
        )
    )

    model = StandardizedTransformer(model_name)
    for alpha in alphas:
        safe_alpha = str(alpha).replace(".", "p")
        cfg_dir = out_root / f"alpha_{safe_alpha}"
        steered_rows = load_existing_rows(cfg_dir / "steered_only.jsonl")
        failures_path = cfg_dir / "failures.json"
        failures = json.loads(failures_path.read_text()) if failures_path.exists() else []

        console.rule(f"item-conditioned oracle layers={layers_label(selected_layers)} alpha={alpha}")
        for persona in personas:
            already = {
                (row.qid, row.condition)
                for row in steered_rows
                if row.persona_id == persona.id
            }
            expected = {
                (qa.qid, condition)
                for qa in qa_by_persona[persona.id]
                for condition in ("steered_own_item", "steered_cross_item")
            }
            if already >= expected:
                console.print(f"[cyan]Skipping alpha={alpha} for {persona.name}: already complete[/]")
                continue
            if already:
                steered_rows = [row for row in steered_rows if row.persona_id != persona.id]

            cross_id = cross_source_for[persona.id]
            persona_rows: list[ChoiceEvalResult] = []
            try:
                for batch_idx, qa_batch in enumerate(
                    chunked(qa_by_persona[persona.id], args.question_batch_size),
                    start=1,
                ):
                    own_vectors = torch.stack(
                        [
                            vector_bank[persona.id][item_key_for_qa(qa, persona.id)]
                            for qa in qa_batch
                        ],
                        dim=0,
                    )
                    cross_vectors = torch.stack(
                        [
                            vector_bank[cross_id][item_key_for_qa(qa, persona.id)]
                            for qa in qa_batch
                        ],
                        dim=0,
                    )

                    own_batch = run_with_remote_retry(
                        lambda qa_batch=qa_batch, own_vectors=own_vectors: score_item_conditioned_batch(
                            model,
                            persona,
                            qa_batch,
                            condition_name="steered_own_item",
                            steering_vectors=own_vectors,
                            steering_layers=steering_layers,
                            steering_alpha=alpha,
                            remote=args.remote,
                        ),
                        label=f"item own eval for {persona.name} alpha={alpha} chunk {batch_idx}",
                    )
                    persona_rows.extend(own_batch)

                    cross_batch = run_with_remote_retry(
                        lambda qa_batch=qa_batch, cross_vectors=cross_vectors: score_item_conditioned_batch(
                            model,
                            persona,
                            qa_batch,
                            condition_name="steered_cross_item",
                            steering_vectors=cross_vectors,
                            steering_layers=steering_layers,
                            steering_alpha=alpha,
                            remote=args.remote,
                        ),
                        label=f"item cross eval for {persona.name} alpha={alpha} chunk {batch_idx}",
                    )
                    persona_rows.extend(cross_batch)
                steered_rows.extend(persona_rows)
            except Exception as exc:
                failures.append(
                    {
                        "persona_id": persona.id,
                        "persona_name": persona.name,
                        "alpha": alpha,
                        "cross_source_id": cross_id,
                        "error": str(exc),
                    }
                )
                console.print(f"[red]Recorded item-conditioned failure for {persona.name}[/]")

            write_config_outputs(
                cfg_dir,
                reference_rows=reference_rows,
                steered_rows=steered_rows,
                metadata={
                    "source_run_root": str(source_root),
                    "model": model_name,
                    "qa_type": args.qa_type,
                    "questions_per_persona": args.questions_per_persona,
                    "question_batch_size": args.question_batch_size,
                    "remote": args.remote,
                    "layers": selected_layers if selected_layers is not None else "all",
                    "alpha": alpha,
                    "cross_source_for": cross_source_for,
                    "negative_variant": source_metadata.get("negative_variant"),
                    "method": "per_item_mean_contrast",
                    "center": source_metadata.get("center"),
                    "vector_pool_persona_count": len(vector_pool_personas),
                },
                failures=failures,
            )
            console.print(f"[green]Checkpointed item-conditioned alpha={alpha} after {persona.name}[/]")


if __name__ == "__main__":
    main()
