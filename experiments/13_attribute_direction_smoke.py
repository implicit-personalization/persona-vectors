#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
from dotenv import load_dotenv
import nnsight
from nnsight.intervention.backends.remote import RemoteBackend
from nnterp import StandardizedTransformer
from persona_data.environment import set_seed
from persona_data.synth_persona import QAPair, SynthPersonaDataset
from rich.console import Console

from persona_vectors.eval import (
    choice_token_ids,
    score_choice_distribution_batched,
)
from persona_vectors.mc_prompt_contract import render_mc_generation_prompt
from persona_vectors.steering_eval_utils import is_oom_error, run_with_remote_retry

console = Console()


@dataclass(frozen=True)
class AttributeProbe:
    qid: str
    question: str
    choices: list[str]
    positive_choice_index: int
    negative_choice_index: int

    def to_qa(self) -> QAPair:
        return QAPair(
            qid=self.qid,
            type="implicit",
            question=self.question,
            answer=self.choices[self.positive_choice_index],
            difficulty=1,
            answer_format="choice",
            choices=self.choices,
            correct_choice_index=self.positive_choice_index,
            tags=["attribute_direction_smoke"],
        )


POLITICAL_VIEW_PROBES = [
    AttributeProbe(
        qid="political_social_safety_net",
        question="Which view sounds most natural to you when thinking about economic policy?",
        choices=[
            "Government should do more to reduce inequality and provide a stronger safety net.",
            "Government should stay smaller, tax less, and leave more decisions to private citizens.",
            "The best answer depends on the specific program and its cost.",
            "Local communities matter more than national ideology here.",
            "Not enough information from the context.",
        ],
        positive_choice_index=0,
        negative_choice_index=1,
    ),
    AttributeProbe(
        qid="political_regulation_business",
        question="When business interests conflict with consumer or worker protections, what instinct do you usually have?",
        choices=[
            "Avoid heavy-handed rules unless there is clear evidence of harm.",
            "Use regulation when markets are producing unfair or unsafe outcomes.",
            "Look for voluntary standards before legal requirements.",
            "Separate small businesses from large corporations before deciding.",
            "Not enough information from the context.",
        ],
        positive_choice_index=1,
        negative_choice_index=0,
    ),
    AttributeProbe(
        qid="political_tradition_change",
        question="How do you tend to think about social change?",
        choices=[
            "Social change is often necessary when old norms exclude or harm people.",
            "Traditions deserve a strong presumption of respect before society changes them.",
            "Some changes are overdue, but the pace and implementation matter.",
            "The question depends more on family and community than politics.",
            "Not enough information from the context.",
        ],
        positive_choice_index=0,
        negative_choice_index=1,
    ),
    AttributeProbe(
        qid="political_public_services",
        question="What sounds closer to your view of public services?",
        choices=[
            "Private and charitable efforts are usually more accountable than expanded government systems.",
            "Public systems are important because basic needs should not depend only on private charity.",
            "Public services should be judged mainly by measurable outcomes.",
            "The right balance depends on whether the service is local or federal.",
            "Not enough information from the context.",
        ],
        positive_choice_index=1,
        negative_choice_index=0,
    ),
    AttributeProbe(
        qid="political_inequality_cause",
        question="When you see major inequality, what explanation feels most plausible at first?",
        choices=[
            "Unequal outcomes often reflect structural barriers and unequal starting points.",
            "Unequal outcomes often reflect differences in choices, effort, and responsibility.",
            "Both structure and personal responsibility matter, and neither should be ignored.",
            "The causes are usually too local and personal for broad ideology.",
            "Not enough information from the context.",
        ],
        positive_choice_index=0,
        negative_choice_index=1,
    ),
    AttributeProbe(
        qid="political_institutional_trust",
        question="Which political instinct fits you better when institutions disappoint people?",
        choices=[
            "Reform institutions so they are more inclusive, accountable, and publicly responsive.",
            "Limit institutional power and protect individual freedom from centralized overreach.",
            "Replace bad leadership before changing the institution itself.",
            "Trust should be rebuilt through local relationships before national politics.",
            "Not enough information from the context.",
        ],
        positive_choice_index=0,
        negative_choice_index=1,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Minimal attribute-level steering smoke test. Builds a political_views "
            "liberal-minus-conservative direction from SynthPersona biography-context "
            "MC probes, then evaluates +direction, -direction, and shuffled-label control "
            "on no-context probes."
        )
    )
    parser.add_argument("--model", default="google/gemma-2-9b-it")
    parser.add_argument("--attribute", default="political_views")
    parser.add_argument(
        "--positive-values",
        default="Liberal,Slightly liberal",
        help="Comma-separated attribute values for the positive class.",
    )
    parser.add_argument(
        "--negative-values",
        default="Conservative,Slightly conservative,Extremely conservative",
        help="Comma-separated attribute values for the negative class.",
    )
    parser.add_argument("--positive-label", default="liberal")
    parser.add_argument("--negative-label", default="conservative")
    parser.add_argument("--layer", type=int, default=41)
    parser.add_argument("--train-per-class", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--alphas", default="0.25,0.5,1.0")
    parser.add_argument("--remote", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--extraction-batch-size", type=int, default=6)
    parser.add_argument("--include-control", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path("/Users/hengxuli/Repos/synth-persona/.env"),
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def parse_csv_values(raw: str) -> set[str]:
    values = {item.strip() for item in raw.split(",") if item.strip()}
    if not values:
        raise ValueError("Expected at least one comma-separated value")
    return values


def parse_alphas(raw: str) -> list[float]:
    alphas = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not alphas:
        raise ValueError("Expected at least one alpha")
    return alphas


def default_output_dir(*, model_name: str, attribute: str, layer: int) -> Path:
    run_id = datetime.now().strftime("%Y%m%dT%H%M%SZ")
    return (
        Path("artifacts")
        / "experiments"
        / "attribute_direction_smoke"
        / f"{run_id}__{model_name.replace('/', '__')}__{attribute}__layer_{layer}"
    )


def select_persona_groups(
    dataset: SynthPersonaDataset,
    *,
    attribute: str,
    positive_values: set[str],
    negative_values: set[str],
    train_per_class: int,
    seed: int,
) -> tuple[list, list, list, list]:
    positives = [
        persona
        for persona in dataset
        if str(persona.persona.get(attribute, "")).strip() in positive_values
    ]
    negatives = [
        persona
        for persona in dataset
        if str(persona.persona.get(attribute, "")).strip() in negative_values
    ]
    rng = random.Random(seed)
    rng.shuffle(positives)
    rng.shuffle(negatives)
    if len(positives) < train_per_class + 1 or len(negatives) < train_per_class + 1:
        raise ValueError(
            f"Need at least train_per_class+1 examples per class; got "
            f"positive={len(positives)}, negative={len(negatives)}"
        )
    train_pos = positives[:train_per_class]
    train_neg = negatives[:train_per_class]
    heldout_pos = positives[train_per_class:]
    heldout_neg = negatives[train_per_class:]
    return train_pos, train_neg, heldout_pos, heldout_neg


def prepare_probe_inputs(
    model: StandardizedTransformer,
    personas: list,
    probes: list[AttributeProbe],
    *,
    attribute: str,
):
    input_ids_list: list[torch.Tensor] = []
    token_masks: list[torch.Tensor] = []
    rows: list[dict] = []
    for persona in personas:
        for probe in probes:
            qa = probe.to_qa()
            prompt, prompt_len = render_mc_generation_prompt(
                model.tokenizer,
                persona=persona,
                qa=qa,
                condition="biography",
            )
            input_ids = model.tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=False,
            ).input_ids[0]
            if int(input_ids.shape[0]) != prompt_len:
                raise ValueError("Prompt length mismatch after rendering")
            mask = torch.zeros_like(input_ids, dtype=torch.bool)
            mask[prompt_len - 1] = True
            input_ids_list.append(input_ids)
            token_masks.append(mask)
            rows.append(
                {
                    "persona_id": persona.id,
                    "persona_name": persona.name,
                    "probe_id": probe.qid,
                    "attribute_value": persona.persona.get(attribute),
                    "prompt_token_count": prompt_len,
                }
            )
    return input_ids_list, token_masks, rows


def extract_layer_activations(
    model: StandardizedTransformer,
    input_ids_list: list[torch.Tensor],
    token_masks: list[torch.Tensor],
    *,
    layer: int,
    remote: bool,
) -> torch.Tensor:
    if len(input_ids_list) != len(token_masks):
        raise ValueError("input_ids_list and token_masks must have the same length")

    masks = [torch.as_tensor(mask, dtype=torch.bool) for mask in token_masks]
    for input_ids, mask in zip(input_ids_list, masks, strict=True):
        if input_ids.ndim != 1:
            raise ValueError(f"expected 1-D input ids, got shape {tuple(input_ids.shape)}")
        if input_ids.shape[0] != mask.shape[0]:
            raise ValueError(
                f"input ids length {input_ids.shape[0]} does not match mask length {mask.shape[0]}"
            )
        if not bool(mask.any()):
            raise ValueError("token_mask selects zero tokens")

    backend = RemoteBackend(model.to_model_key()) if remote else None
    with torch.no_grad(), model.session(remote=remote, backend=backend):
        all_hs: list[torch.Tensor] = nnsight.save([])
        for input_ids, mask in zip(input_ids_list, masks, strict=True):
            with model.trace(input_ids.unsqueeze(0)) as tracer:
                mask_on_device = mask.to(device=model.layers_output[layer].device)
                layer_mean = model.layers_output[layer][0, mask_on_device].mean(dim=0)
                saved = nnsight.save(layer_mean.detach().cpu())
                tracer.stop()
            all_hs.append(saved)

    return torch.stack(all_hs, dim=0)


def extract_probe_layer_activations(
    model: StandardizedTransformer,
    *,
    input_ids_list: list[torch.Tensor],
    token_masks: list[torch.Tensor],
    rows: list[dict],
    layer: int,
    remote: bool,
    batch_size: int,
    label: str,
) -> tuple[torch.Tensor, list[dict], list[dict]]:
    chunks: list[torch.Tensor] = []
    kept_rows: list[dict] = []
    skipped_rows: list[dict] = []
    oom_persona_ids: set[str] = set()
    for start in range(0, len(input_ids_list), batch_size):
        end = min(start + batch_size, len(input_ids_list))
        if end - start == 1 and rows[start]["persona_id"] in oom_persona_ids:
            skipped_rows.append(
                {
                    **rows[start],
                    "skip_reason": "persona_previously_hit_remote_oom",
                    "error_type": None,
                    "error": None,
                }
            )
            continue
        try:
            chunk_vectors = run_with_remote_retry(
                lambda start=start, end=end: extract_layer_activations(
                    model,
                    input_ids_list=input_ids_list[start:end],
                    token_masks=token_masks[start:end],
                    layer=layer,
                    remote=remote,
                ),
                label=f"{label} extraction {start + 1}-{end}",
                retries=2,
                sleep_seconds=5,
            )
        except Exception as exc:
            if is_oom_error(exc) and end - start == 1:
                oom_persona_ids.add(rows[start]["persona_id"])
                skipped = {
                    **rows[start],
                    "skip_reason": "remote_oom",
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:500],
                }
                skipped_rows.append(skipped)
                console.print(
                    f"[yellow]Skipping {label} extraction {start + 1}-{end} "
                    f"after repeated remote OOM: {skipped['persona_id']} "
                    f"{skipped['probe_id']}[/]"
                )
                continue
            raise
        chunks.append(chunk_vectors)
        kept_rows.extend(rows[start:end])

    if not chunks:
        raise RuntimeError(f"No activations extracted for {label}")
    return torch.cat(chunks, dim=0), kept_rows, skipped_rows


def make_vector(
    acts: torch.Tensor,
    rows: list[dict],
    *,
    positive_ids: set[str],
    negative_ids: set[str],
) -> torch.Tensor:
    pos_indices = [idx for idx, row in enumerate(rows) if row["persona_id"] in positive_ids]
    neg_indices = [idx for idx, row in enumerate(rows) if row["persona_id"] in negative_ids]
    if not pos_indices or not neg_indices:
        raise ValueError("Cannot build vector without both positive and negative rows")
    return acts[pos_indices, :].float().mean(dim=0) - acts[neg_indices, :].float().mean(dim=0)


def score_bare_probes(
    model: StandardizedTransformer,
    *,
    probes: list[AttributeProbe],
    reference_persona,
    remote: bool,
    steering_layer: int | None = None,
    steering_vector: torch.Tensor | None = None,
    steering_alpha: float | None = None,
) -> list[dict]:
    prompts: list[str] = []
    prompt_lens: list[int] = []
    choice_ids_list: list[list[int]] = []
    choice_letters_list: list[list[str]] = []
    for probe in probes:
        qa = probe.to_qa()
        prompt, prompt_len = render_mc_generation_prompt(
            model.tokenizer,
            persona=reference_persona,
            qa=qa,
            condition="bare",
        )
        choice_letters, choice_ids = choice_token_ids(model.tokenizer, qa)
        prompts.append(prompt)
        prompt_lens.append(prompt_len)
        choice_ids_list.append(choice_ids)
        choice_letters_list.append(choice_letters)

    logprobs, probs = score_choice_distribution_batched(
        model,
        prompts,
        prompt_lens,
        choice_ids_list,
        remote=remote,
        steering_layer=steering_layer,
        steering_vector=steering_vector,
        steering_alpha=steering_alpha,
    )
    rows: list[dict] = []
    for probe, letters, lp, prob in zip(probes, choice_letters_list, logprobs, probs, strict=True):
        pred_idx = int(lp.argmax().item())
        positive_prob = float(prob[probe.positive_choice_index].item())
        negative_prob = float(prob[probe.negative_choice_index].item())
        positive_logprob = float(lp[probe.positive_choice_index].item())
        negative_logprob = float(lp[probe.negative_choice_index].item())
        rows.append(
            {
                "probe_id": probe.qid,
                "question": probe.question,
                "positive_choice_letter": letters[probe.positive_choice_index],
                "negative_choice_letter": letters[probe.negative_choice_index],
                "predicted_letter": letters[pred_idx],
                "positive_prob": positive_prob,
                "negative_prob": negative_prob,
                "positive_minus_negative_prob": positive_prob - negative_prob,
                "positive_logprob": positive_logprob,
                "negative_logprob": negative_logprob,
                "positive_minus_negative_logprob": positive_logprob - negative_logprob,
                "choice_letters": letters,
                "choice_probs": [float(value) for value in prob.tolist()],
                "choice_logprobs": [float(value) for value in lp.tolist()],
            }
        )
    return rows


def summarize_score_rows(rows: list[dict]) -> dict:
    by_condition: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_condition[row["condition"]].append(row)

    bare_by_probe = {
        row["probe_id"]: row
        for row in by_condition.get("bare", [])
    }
    summary: dict[str, dict] = {}
    for condition, items in sorted(by_condition.items()):
        n = len(items)
        summary[condition] = {
            "n_examples": n,
            "mean_positive_prob": sum(row["positive_prob"] for row in items) / n if n else 0.0,
            "mean_negative_prob": sum(row["negative_prob"] for row in items) / n if n else 0.0,
            "mean_positive_minus_negative_prob": (
                sum(row["positive_minus_negative_prob"] for row in items) / n if n else 0.0
            ),
            "mean_positive_logprob": (
                sum(row["positive_logprob"] for row in items) / n if n else 0.0
            ),
            "mean_negative_logprob": (
                sum(row["negative_logprob"] for row in items) / n if n else 0.0
            ),
            "mean_positive_minus_negative_logprob": (
                sum(row["positive_minus_negative_logprob"] for row in items) / n
                if n
                else 0.0
            ),
            "predicted_letter_counts": dict(Counter(row["predicted_letter"] for row in items)),
        }
        if condition != "bare" and bare_by_probe:
            comparable = [
                (row, bare_by_probe[row["probe_id"]])
                for row in items
                if row["probe_id"] in bare_by_probe
            ]
            if comparable:
                summary[f"{condition}_vs_bare"] = {
                    "n_examples": len(comparable),
                    "mean_delta_positive_prob": sum(
                        row["positive_prob"] - bare["positive_prob"]
                        for row, bare in comparable
                    )
                    / len(comparable),
                    "mean_delta_negative_prob": sum(
                        row["negative_prob"] - bare["negative_prob"]
                        for row, bare in comparable
                    )
                    / len(comparable),
                    "mean_delta_positive_minus_negative_prob": sum(
                        row["positive_minus_negative_prob"]
                        - bare["positive_minus_negative_prob"]
                        for row, bare in comparable
                    )
                    / len(comparable),
                    "mean_delta_positive_minus_negative_logprob": sum(
                        row["positive_minus_negative_logprob"]
                        - bare["positive_minus_negative_logprob"]
                        for row, bare in comparable
                    )
                    / len(comparable),
                }
    return summary


def projection_rows(
    *,
    acts: torch.Tensor,
    rows: list[dict],
    vector: torch.Tensor,
    split: str,
    positive_ids: set[str],
    negative_ids: set[str],
) -> list[dict]:
    unit = vector.float() / (vector.float().norm() + 1e-8)
    output: list[dict] = []
    for idx, row in enumerate(rows):
        if row["persona_id"] in positive_ids:
            label = "positive"
        elif row["persona_id"] in negative_ids:
            label = "negative"
        else:
            label = "excluded"
        output.append(
            {
                **row,
                "split": split,
                "label": label,
                "projection": float(acts[idx, :].float().dot(unit).item()),
            }
        )
    return output


def summarize_projection(rows: list[dict]) -> dict:
    by_key: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        by_key[(row["split"], row["label"])].append(row["projection"])
    return {
        f"{split}::{label}": {
            "n": len(values),
            "mean_projection": sum(values) / len(values) if values else 0.0,
        }
        for (split, label), values in sorted(by_key.items())
    }


def main() -> None:
    args = parse_args()
    load_dotenv()
    if args.env_file is not None:
        load_dotenv(args.env_file, override=False)
    set_seed(args.seed)

    if args.remote and not os.environ.get("NDIF_API_KEY"):
        raise RuntimeError("NDIF_API_KEY is required for remote runs.")

    positive_values = parse_csv_values(args.positive_values)
    negative_values = parse_csv_values(args.negative_values)
    alphas = parse_alphas(args.alphas)
    out_dir = args.out_dir or default_output_dir(
        model_name=args.model,
        attribute=args.attribute,
        layer=args.layer,
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = SynthPersonaDataset()
    train_pos, train_neg, heldout_pos, heldout_neg = select_persona_groups(
        dataset,
        attribute=args.attribute,
        positive_values=positive_values,
        negative_values=negative_values,
        train_per_class=args.train_per_class,
        seed=args.seed,
    )
    train_personas = train_pos + train_neg
    heldout_personas = heldout_pos + heldout_neg
    train_pos_ids = {persona.id for persona in train_pos}
    train_neg_ids = {persona.id for persona in train_neg}
    heldout_pos_ids = {persona.id for persona in heldout_pos}
    heldout_neg_ids = {persona.id for persona in heldout_neg}

    model = StandardizedTransformer(args.model)
    probes = POLITICAL_VIEW_PROBES
    if args.attribute != "political_views":
        raise ValueError("This first smoke currently only defines political_views probes")

    train_ids, train_masks, train_rows = prepare_probe_inputs(
        model,
        train_personas,
        probes,
        attribute=args.attribute,
    )
    heldout_ids, heldout_masks, heldout_rows = prepare_probe_inputs(
        model,
        heldout_personas,
        probes,
        attribute=args.attribute,
    )

    console.print(
        f"[cyan]Extracting layer {args.layer} train activations: {len(train_ids)} prompts; "
        f"heldout activations: {len(heldout_ids)} prompts[/]"
    )
    train_acts, train_rows, skipped_train_rows = extract_probe_layer_activations(
        model,
        input_ids_list=train_ids,
        token_masks=train_masks,
        rows=train_rows,
        layer=args.layer,
        remote=args.remote,
        batch_size=args.extraction_batch_size,
        label="attribute train",
    )
    heldout_acts, heldout_rows, skipped_heldout_rows = extract_probe_layer_activations(
        model,
        input_ids_list=heldout_ids,
        token_masks=heldout_masks,
        rows=heldout_rows,
        layer=args.layer,
        remote=args.remote,
        batch_size=args.extraction_batch_size,
        label="attribute heldout",
    )

    attribute_vector = make_vector(
        train_acts,
        train_rows,
        positive_ids=train_pos_ids,
        negative_ids=train_neg_ids,
    )
    control_vector = None
    if args.include_control:
        rng = random.Random(args.seed + 17)
        train_ids_all = [persona.id for persona in train_personas]
        shuffled_positive = set(rng.sample(train_ids_all, k=len(train_pos_ids)))
        shuffled_negative = set(train_ids_all) - shuffled_positive
        control_vector = make_vector(
            train_acts,
            train_rows,
            positive_ids=shuffled_positive,
            negative_ids=shuffled_negative,
        )
    else:
        shuffled_positive = set()
        shuffled_negative = set()

    torch.save(
        {
            "attribute_vector": attribute_vector.detach().cpu(),
            "control_vector": control_vector.detach().cpu() if control_vector is not None else None,
            "layer": args.layer,
            "attribute": args.attribute,
        },
        out_dir / "vectors.pt",
    )

    reference_persona = next(iter(dataset))
    score_rows: list[dict] = []
    bare_rows = run_with_remote_retry(
        lambda: score_bare_probes(
            model,
            probes=probes,
            reference_persona=reference_persona,
            remote=args.remote,
        ),
        label="bare attribute probes",
        retries=5,
        sleep_seconds=10,
    )
    for row in bare_rows:
        score_rows.append({**row, "condition": "bare", "alpha": 0.0})

    for alpha in alphas:
        for condition, vector, signed_alpha in [
            ("steer_positive", attribute_vector, alpha),
            ("steer_negative", attribute_vector, -alpha),
        ]:
            rows = run_with_remote_retry(
                lambda vector=vector, signed_alpha=signed_alpha: score_bare_probes(
                    model,
                    probes=probes,
                    reference_persona=reference_persona,
                    remote=args.remote,
                    steering_layer=args.layer,
                    steering_vector=vector,
                    steering_alpha=signed_alpha,
                ),
                label=f"{condition} alpha={alpha}",
                retries=5,
                sleep_seconds=10,
            )
            for row in rows:
                score_rows.append({**row, "condition": condition, "alpha": alpha})

        if control_vector is not None:
            rows = run_with_remote_retry(
                lambda alpha=alpha: score_bare_probes(
                    model,
                    probes=probes,
                    reference_persona=reference_persona,
                    remote=args.remote,
                    steering_layer=args.layer,
                    steering_vector=control_vector,
                    steering_alpha=alpha,
                ),
                label=f"control_positive alpha={alpha}",
                retries=5,
                sleep_seconds=10,
            )
            for row in rows:
                score_rows.append({**row, "condition": "control_positive", "alpha": alpha})

    projection = []
    projection.extend(
        projection_rows(
            acts=train_acts,
            rows=train_rows,
            vector=attribute_vector,
            split="train",
            positive_ids=train_pos_ids,
            negative_ids=train_neg_ids,
        )
    )
    projection.extend(
        projection_rows(
            acts=heldout_acts,
            rows=heldout_rows,
            vector=attribute_vector,
            split="heldout",
            positive_ids=heldout_pos_ids,
            negative_ids=heldout_neg_ids,
        )
    )

    metadata = {
        "model": args.model,
        "attribute": args.attribute,
        "positive_values": sorted(positive_values),
        "negative_values": sorted(negative_values),
        "positive_label": args.positive_label,
        "negative_label": args.negative_label,
        "layer": args.layer,
        "alphas": alphas,
        "train_per_class": args.train_per_class,
        "train_positive": [
            {"id": persona.id, "name": persona.name, "value": persona.persona[args.attribute]}
            for persona in train_pos
        ],
        "train_negative": [
            {"id": persona.id, "name": persona.name, "value": persona.persona[args.attribute]}
            for persona in train_neg
        ],
        "heldout_positive_count": len(heldout_pos),
        "heldout_negative_count": len(heldout_neg),
        "probe_count": len(probes),
        "kept_train_prompt_count": len(train_rows),
        "kept_heldout_prompt_count": len(heldout_rows),
        "skipped_extraction_count": len(skipped_train_rows) + len(skipped_heldout_rows),
        "skipped_train_extractions": skipped_train_rows,
        "skipped_heldout_extractions": skipped_heldout_rows,
        "attribute_vector_norm": float(attribute_vector.norm().item()),
        "control_vector_norm": float(control_vector.norm().item()) if control_vector is not None else None,
        "shuffled_positive_ids": sorted(shuffled_positive),
        "shuffled_negative_ids": sorted(shuffled_negative),
    }
    summary = {
        "score_summary": summarize_score_rows(score_rows),
        "projection_summary": summarize_projection(projection),
    }

    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (out_dir / "summary.json").write_text(
        json.dumps({"metadata": metadata, **summary}, indent=2)
    )
    with (out_dir / "per_probe.jsonl").open("w") as handle:
        for row in score_rows:
            handle.write(json.dumps(row) + "\n")
    with (out_dir / "projection.jsonl").open("w") as handle:
        for row in projection:
            handle.write(json.dumps(row) + "\n")

    console.print_json(json.dumps(summary))
    console.print(f"[green]Wrote attribute direction smoke to {out_dir}[/]")


if __name__ == "__main__":
    main()
