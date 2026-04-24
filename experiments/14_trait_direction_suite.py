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

import nnsight
import torch
from dotenv import load_dotenv
from nnterp import StandardizedTransformer
from nnsight.intervention.backends.remote import RemoteBackend
from persona_data.environment import set_seed
from persona_data.synth_persona import QAPair, SynthPersonaDataset
from rich.console import Console

from persona_vectors.eval import choice_token_ids, score_choice_distribution_batched
from persona_vectors.mc_prompt_contract import (
    render_mc_generation_prompt,
    render_mc_prompt_with_answer,
)
from persona_vectors.steering_eval_utils import is_oom_error, run_with_remote_retry

console = Console()


@dataclass(frozen=True)
class AttributeProbe:
    qid: str
    question: str
    choices: tuple[str, str, str, str, str]
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
            choices=list(self.choices),
            correct_choice_index=self.positive_choice_index,
            tags=["trait_direction_suite"],
        )


def rotate_substantive_options(probe: AttributeProbe, rotation: int) -> AttributeProbe:
    """Rotate A-D choices while keeping E as the no-context fallback option."""
    rotation = rotation % 4
    if rotation == 0:
        return probe
    if probe.positive_choice_index >= 4 or probe.negative_choice_index >= 4:
        raise ValueError("Option rotation expects positive/negative choices inside A-D")
    substantive = list(probe.choices[:4])
    rotated = substantive[rotation:] + substantive[:rotation]
    return AttributeProbe(
        qid=f"{probe.qid}__rot{rotation}",
        question=probe.question,
        choices=tuple(rotated + [probe.choices[4]]),
        positive_choice_index=(probe.positive_choice_index - rotation) % 4,
        negative_choice_index=(probe.negative_choice_index - rotation) % 4,
    )


@dataclass(frozen=True)
class TraitSpec:
    name: str
    attribute: str
    positive_label: str
    negative_label: str
    positive_values: tuple[str, ...]
    negative_values: tuple[str, ...]
    probes: tuple[AttributeProbe, ...]


POLITICAL_VIEW_PROBES = (
    AttributeProbe(
        qid="political_social_safety_net",
        question="Which view sounds most natural to you when thinking about economic policy?",
        choices=(
            "Government should do more to reduce inequality and provide a stronger safety net.",
            "Government should stay smaller, tax less, and leave more decisions to private citizens.",
            "The best answer depends on the specific program and its cost.",
            "Local communities matter more than national ideology here.",
            "Not enough information from the context.",
        ),
        positive_choice_index=0,
        negative_choice_index=1,
    ),
    AttributeProbe(
        qid="political_regulation_business",
        question="When business interests conflict with consumer or worker protections, what instinct do you usually have?",
        choices=(
            "Avoid heavy-handed rules unless there is clear evidence of harm.",
            "Use regulation when markets are producing unfair or unsafe outcomes.",
            "Look for voluntary standards before legal requirements.",
            "Separate small businesses from large corporations before deciding.",
            "Not enough information from the context.",
        ),
        positive_choice_index=1,
        negative_choice_index=0,
    ),
    AttributeProbe(
        qid="political_tradition_change",
        question="How do you tend to think about social change?",
        choices=(
            "Social change is often necessary when old norms exclude or harm people.",
            "Traditions deserve a strong presumption of respect before society changes them.",
            "Some changes are overdue, but the pace and implementation matter.",
            "The question depends more on family and community than politics.",
            "Not enough information from the context.",
        ),
        positive_choice_index=0,
        negative_choice_index=1,
    ),
    AttributeProbe(
        qid="political_public_services",
        question="What sounds closer to your view of public services?",
        choices=(
            "Private and charitable efforts are usually more accountable than expanded government systems.",
            "Public systems are important because basic needs should not depend only on private charity.",
            "Public services should be judged mainly by measurable outcomes.",
            "The right balance depends on whether the service is local or federal.",
            "Not enough information from the context.",
        ),
        positive_choice_index=1,
        negative_choice_index=0,
    ),
    AttributeProbe(
        qid="political_inequality_cause",
        question="When you see major inequality, what explanation feels most plausible at first?",
        choices=(
            "Unequal outcomes often reflect structural barriers and unequal starting points.",
            "Unequal outcomes often reflect differences in choices, effort, and responsibility.",
            "Both structure and personal responsibility matter, and neither should be ignored.",
            "The causes are usually too local and personal for broad ideology.",
            "Not enough information from the context.",
        ),
        positive_choice_index=0,
        negative_choice_index=1,
    ),
    AttributeProbe(
        qid="political_institutional_trust",
        question="Which political instinct fits you better when institutions disappoint people?",
        choices=(
            "Reform institutions so they are more inclusive, accountable, and publicly responsive.",
            "Limit institutional power and protect individual freedom from centralized overreach.",
            "Replace bad leadership before changing the institution itself.",
            "Trust should be rebuilt through local relationships before national politics.",
            "Not enough information from the context.",
        ),
        positive_choice_index=0,
        negative_choice_index=1,
    ),
)


TOTAL_WEALTH_PROBES = (
    AttributeProbe(
        qid="wealth_emergency_repair",
        question="If your car needed an unexpected repair this month, which reaction sounds closer to your situation?",
        choices=(
            "I could probably cover it from savings without changing much else.",
            "I would need to postpone bills, borrow, or cut spending sharply.",
            "I would compare repair quotes before deciding what to do.",
            "I would ask someone I trust to help judge whether the repair is necessary.",
            "Not enough information from the context.",
        ),
        positive_choice_index=0,
        negative_choice_index=1,
    ),
    AttributeProbe(
        qid="wealth_housing_repair",
        question="If something important broke at home, what would your first practical constraint be?",
        choices=(
            "Finding the cheapest temporary fix so the cost does not get out of hand.",
            "Paying for a durable repair if it prevents bigger costs later.",
            "Figuring out whether insurance or a warranty applies.",
            "Asking a neighbor or relative who has handled the same problem.",
            "Not enough information from the context.",
        ),
        positive_choice_index=1,
        negative_choice_index=0,
    ),
    AttributeProbe(
        qid="wealth_job_loss_buffer",
        question="How would a short interruption in income most likely feel?",
        choices=(
            "It would become urgent quickly because the margin is thin.",
            "It would be annoying, but mostly a scheduling problem.",
            "There is enough cushion that it would not immediately threaten basics.",
            "The answer depends more on health and family obligations than money.",
            "Not enough information from the context.",
        ),
        positive_choice_index=2,
        negative_choice_index=0,
    ),
    AttributeProbe(
        qid="wealth_helping_relative",
        question="If a close relative needed a small financial favor, what sounds most realistic?",
        choices=(
            "I would offer time or advice because spare cash is limited.",
            "I would first ask exactly what the money is for.",
            "I would help only if there were a clear repayment plan.",
            "I could help somewhat without putting my own basics at risk.",
            "Not enough information from the context.",
        ),
        positive_choice_index=3,
        negative_choice_index=0,
    ),
    AttributeProbe(
        qid="wealth_purchase_tradeoff",
        question="When buying something important, which tradeoff feels more natural?",
        choices=(
            "Paying more upfront for quality can make sense if it lasts longer.",
            "Keeping the upfront cost low matters most because cash should stay available.",
            "I usually delay the purchase until there is a sale.",
            "I mostly rely on reviews and recommendations before choosing.",
            "Not enough information from the context.",
        ),
        positive_choice_index=0,
        negative_choice_index=1,
    ),
    AttributeProbe(
        qid="wealth_medical_bill",
        question="If a moderate medical bill arrived unexpectedly, which answer sounds closer?",
        choices=(
            "A payment plan or delay would probably be necessary.",
            "I could absorb at least a moderate bill without immediate crisis.",
            "I would first challenge the bill or ask for an itemized statement.",
            "I would ask whether the provider has a financial-assistance policy.",
            "Not enough information from the context.",
        ),
        positive_choice_index=1,
        negative_choice_index=0,
    ),
)


TRAIT_SPECS: dict[str, TraitSpec] = {
    "political_views": TraitSpec(
        name="political_views",
        attribute="political_views",
        positive_label="liberal",
        negative_label="conservative",
        positive_values=("Liberal", "Slightly liberal"),
        negative_values=("Conservative", "Slightly conservative", "Extremely conservative"),
        probes=POLITICAL_VIEW_PROBES,
    ),
    "total_wealth": TraitSpec(
        name="total_wealth",
        attribute="total_wealth",
        positive_label="higher_wealth",
        negative_label="lower_wealth",
        positive_values=(
            "$40,000 to $75,000",
            "$75,000 to $100,000",
            "$100,000 to $150,000",
            "$250,000 to $500,000",
        ),
        negative_values=(
            "Less than $5,000",
            "$5,000 to $20,000",
            "$20,000 to $40,000",
        ),
        probes=TOTAL_WEALTH_PROBES,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Trait-level steering suite. Builds ground-truth attribute directions "
            "from SynthPersona seed metadata, then evaluates true, reverse, and "
            "shuffled-label controls on no-context MC probes."
        )
    )
    parser.add_argument("--model", default="google/gemma-2-9b-it")
    parser.add_argument("--traits", default="political_views,total_wealth")
    parser.add_argument(
        "--eval-traits",
        default="same",
        help="'same', 'all', or a comma-separated list of trait probe banks.",
    )
    parser.add_argument(
        "--activation-source",
        choices=["prompt_last", "teacher_forced_choice_response_mean"],
        default="prompt_last",
        help=(
            "prompt_last uses the final prompt token before the answer. "
            "teacher_forced_choice_response_mean averages the forced answer-letter "
            "response tokens and is a diagnostic control, not a final method."
        ),
    )
    parser.add_argument("--layer", type=int, default=41)
    parser.add_argument("--train-per-class", type=int, default=4)
    parser.add_argument("--seeds", default="1337")
    parser.add_argument("--alphas", default="0.25,0.5,1.0")
    parser.add_argument(
        "--eval-option-rotations",
        default="0",
        help=(
            "Comma-separated rotations of substantive A-D options used only during "
            "no-context MC scoring. E stays fixed as the no-context fallback."
        ),
    )
    parser.add_argument("--remote", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--extraction-batch-size", type=int, default=1)
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path("/Users/hengxuli/Repos/synth-persona/.env"),
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def parse_csv(raw: str) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one comma-separated value")
    return values


def parse_int_csv(raw: str) -> list[int]:
    return [int(item) for item in parse_csv(raw)]


def parse_float_csv(raw: str) -> list[float]:
    return [float(item) for item in parse_csv(raw)]


def default_output_dir(*, model_name: str, layer: int, activation_source: str) -> Path:
    run_id = datetime.now().strftime("%Y%m%dT%H%M%SZ")
    return (
        Path("artifacts")
        / "experiments"
        / "trait_direction_suite"
        / f"{run_id}__{model_name.replace('/', '__')}__layer_{layer}__{activation_source}"
    )


def resolve_eval_traits(train_traits: list[str], raw: str) -> dict[str, list[str]]:
    if raw == "same":
        return {trait: [trait] for trait in train_traits}
    if raw == "all":
        all_traits = sorted(TRAIT_SPECS)
        return {trait: all_traits for trait in train_traits}
    requested = parse_csv(raw)
    missing = sorted(set(requested) - set(TRAIT_SPECS))
    if missing:
        raise ValueError(f"Unknown eval traits: {missing}")
    return {trait: requested for trait in train_traits}


def persona_value(persona, attribute: str) -> str:
    return str(persona.persona.get(attribute, "")).strip()


def select_persona_groups(
    dataset: SynthPersonaDataset,
    spec: TraitSpec,
    *,
    train_per_class: int,
    seed: int,
) -> tuple[list, list, list, list]:
    positive_values = set(spec.positive_values)
    negative_values = set(spec.negative_values)
    positives = [
        persona
        for persona in dataset
        if persona_value(persona, spec.attribute) in positive_values
    ]
    negatives = [
        persona
        for persona in dataset
        if persona_value(persona, spec.attribute) in negative_values
    ]
    rng = random.Random(seed)
    rng.shuffle(positives)
    rng.shuffle(negatives)
    if len(positives) < train_per_class + 1 or len(negatives) < train_per_class + 1:
        raise ValueError(
            f"{spec.name} needs train_per_class+1 examples in each class; "
            f"got positive={len(positives)}, negative={len(negatives)}"
        )
    train_pos = positives[:train_per_class]
    train_neg = negatives[:train_per_class]
    heldout_pos = positives[train_per_class:]
    heldout_neg = negatives[train_per_class:]
    return train_pos, train_neg, heldout_pos, heldout_neg


def _find_response_end(input_ids: torch.Tensor, answer_start: int, special_ids: set[int]) -> int:
    for idx in range(answer_start, int(input_ids.shape[0])):
        if int(input_ids[idx]) in special_ids:
            return idx
    return int(input_ids.shape[0])


def prepare_probe_inputs(
    model: StandardizedTransformer,
    personas: list,
    spec: TraitSpec,
    *,
    class_by_persona_id: dict[str, str],
    activation_source: str,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[dict]]:
    input_ids_list: list[torch.Tensor] = []
    token_masks: list[torch.Tensor] = []
    rows: list[dict] = []
    special_ids = set(model.tokenizer.all_special_ids)

    for persona in personas:
        class_label = class_by_persona_id[persona.id]
        for probe in spec.probes:
            qa = probe.to_qa()
            if activation_source == "prompt_last":
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
                answer_letter = None
            elif activation_source == "teacher_forced_choice_response_mean":
                answer_idx = (
                    probe.positive_choice_index
                    if class_label == "positive"
                    else probe.negative_choice_index
                )
                answer_letter = chr(ord("A") + answer_idx)
                full_prompt, answer_start = render_mc_prompt_with_answer(
                    model.tokenizer,
                    persona=persona,
                    qa=qa,
                    condition="biography",
                    answer=answer_letter,
                )
                input_ids = model.tokenizer(
                    full_prompt,
                    return_tensors="pt",
                    add_special_tokens=False,
                ).input_ids[0]
                answer_end = _find_response_end(input_ids, answer_start, special_ids)
                if answer_end <= answer_start:
                    raise ValueError("Teacher-forced answer span is empty")
                mask = torch.zeros_like(input_ids, dtype=torch.bool)
                mask[answer_start:answer_end] = True
            else:
                raise AssertionError(f"Unhandled activation source: {activation_source}")

            input_ids_list.append(input_ids)
            token_masks.append(mask)
            rows.append(
                {
                    "trait": spec.name,
                    "attribute": spec.attribute,
                    "persona_id": persona.id,
                    "persona_name": persona.name,
                    "class_label": class_label,
                    "attribute_value": persona.persona.get(spec.attribute),
                    "probe_id": probe.qid,
                    "activation_source": activation_source,
                    "answer_letter": answer_letter,
                    "masked_token_count": int(mask.sum().item()),
                    "prompt_token_count": int(input_ids.shape[0]),
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


def score_probe_bank(
    model: StandardizedTransformer,
    *,
    spec: TraitSpec,
    reference_persona,
    remote: bool,
    option_rotations: list[int],
    steering_layer: int | None = None,
    steering_vector: torch.Tensor | None = None,
    steering_alpha: float | None = None,
) -> list[dict]:
    prompts: list[str] = []
    prompt_lens: list[int] = []
    choice_ids_list: list[list[int]] = []
    choice_letters_list: list[list[str]] = []

    eval_probes: list[tuple[int, AttributeProbe]] = []
    for rotation in option_rotations:
        for probe in spec.probes:
            eval_probes.append((rotation % 4, rotate_substantive_options(probe, rotation)))

    for _, probe in eval_probes:
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
    for (rotation, probe), letters, lp, prob in zip(
        eval_probes,
        choice_letters_list,
        logprobs,
        probs,
        strict=True,
    ):
        pred_idx = int(lp.argmax().item())
        positive_logprob = float(lp[probe.positive_choice_index].item())
        negative_logprob = float(lp[probe.negative_choice_index].item())
        rows.append(
            {
                "eval_trait": spec.name,
                "probe_id": probe.qid,
                "base_probe_id": probe.qid.split("__rot", maxsplit=1)[0],
                "option_rotation": rotation,
                "question": probe.question,
                "positive_choice_letter": letters[probe.positive_choice_index],
                "negative_choice_letter": letters[probe.negative_choice_index],
                "predicted_letter": letters[pred_idx],
                "positive_prob": float(prob[probe.positive_choice_index].item()),
                "negative_prob": float(prob[probe.negative_choice_index].item()),
                "positive_minus_negative_prob": (
                    float(prob[probe.positive_choice_index].item())
                    - float(prob[probe.negative_choice_index].item())
                ),
                "positive_logprob": positive_logprob,
                "negative_logprob": negative_logprob,
                "positive_minus_negative_logprob": positive_logprob - negative_logprob,
                "choice_letters": letters,
                "choice_probs": [float(value) for value in prob.tolist()],
                "choice_logprobs": [float(value) for value in lp.tolist()],
            }
        )
    return rows


def projection_rows(
    *,
    acts: torch.Tensor,
    rows: list[dict],
    vector: torch.Tensor,
    split: str,
) -> list[dict]:
    unit = vector.float() / (vector.float().norm() + 1e-8)
    return [
        {
            **row,
            "split": split,
            "projection": float(acts[idx, :].float().dot(unit).item()),
        }
        for idx, row in enumerate(rows)
    ]


def summarize_scores(score_rows: list[dict]) -> dict:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in score_rows:
        key = (
            row["train_trait"],
            row["eval_trait"],
            row["seed"],
            row["condition"],
            row["alpha"],
        )
        grouped[key].append(row)

    bare_lookup: dict[tuple, float] = {}
    summary: dict[str, dict] = {}
    for key, rows in sorted(grouped.items()):
        train_trait, eval_trait, seed, condition, alpha = key
        mean_margin = sum(row["positive_minus_negative_logprob"] for row in rows) / len(rows)
        summary_key = (
            f"train={train_trait}::eval={eval_trait}::seed={seed}"
            f"::condition={condition}::alpha={alpha}"
        )
        if condition == "bare":
            bare_lookup[(train_trait, eval_trait, seed)] = mean_margin
        summary[summary_key] = {
            "n_examples": len(rows),
            "mean_positive_minus_negative_logprob": mean_margin,
            "mean_positive_minus_negative_prob": (
                sum(row["positive_minus_negative_prob"] for row in rows) / len(rows)
            ),
            "predicted_letter_counts": dict(Counter(row["predicted_letter"] for row in rows)),
        }

    for key, rows in sorted(grouped.items()):
        train_trait, eval_trait, seed, condition, alpha = key
        if condition == "bare":
            continue
        bare_margin = bare_lookup.get((train_trait, eval_trait, seed))
        if bare_margin is None:
            continue
        mean_margin = sum(row["positive_minus_negative_logprob"] for row in rows) / len(rows)
        summary_key = (
            f"train={train_trait}::eval={eval_trait}::seed={seed}"
            f"::condition={condition}::alpha={alpha}"
        )
        summary[summary_key]["delta_vs_bare_positive_minus_negative_logprob"] = (
            mean_margin - bare_margin
        )

    return summary


def summarize_projection(rows: list[dict]) -> dict:
    grouped: dict[tuple[str, str, str, int], list[float]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["trait"],
                row["split"],
                row["class_label"],
                row["seed"],
            )
        ].append(row["projection"])
    return {
        f"trait={trait}::split={split}::class={label}::seed={seed}": {
            "n": len(values),
            "mean_projection": sum(values) / len(values),
        }
        for (trait, split, label, seed), values in sorted(grouped.items())
    }


def main() -> None:
    args = parse_args()
    load_dotenv()
    if args.env_file is not None:
        load_dotenv(args.env_file, override=False)
    if args.remote and not os.environ.get("NDIF_API_KEY"):
        raise RuntimeError("NDIF_API_KEY is required for remote runs.")

    train_trait_names = parse_csv(args.traits)
    missing_traits = sorted(set(train_trait_names) - set(TRAIT_SPECS))
    if missing_traits:
        raise ValueError(f"Unknown traits: {missing_traits}")
    eval_traits_for_train = resolve_eval_traits(train_trait_names, args.eval_traits)
    seeds = parse_int_csv(args.seeds)
    alphas = parse_float_csv(args.alphas)
    eval_option_rotations = parse_int_csv(args.eval_option_rotations)
    invalid_rotations = [value for value in eval_option_rotations if value < 0 or value > 3]
    if invalid_rotations:
        raise ValueError(f"--eval-option-rotations must be in [0, 3], got {invalid_rotations}")
    out_dir = args.out_dir or default_output_dir(
        model_name=args.model,
        layer=args.layer,
        activation_source=args.activation_source,
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(seeds[0])
    dataset = SynthPersonaDataset()
    model = StandardizedTransformer(args.model)
    reference_persona = next(iter(dataset))

    score_rows: list[dict] = []
    projection: list[dict] = []
    skipped_extractions: list[dict] = []
    vector_payload: dict[str, dict] = {}
    run_metadata: dict = {
        "model": args.model,
        "layer": args.layer,
        "activation_source": args.activation_source,
        "activation_source_note": (
            "teacher_forced_choice_response_mean includes forced MC answer letters and "
            "is only a diagnostic control; prompt_last is the cleaner answer-time vector."
            if args.activation_source == "teacher_forced_choice_response_mean"
            else "prompt_last extracts the final prompt token before the model answer."
        ),
        "train_traits": train_trait_names,
        "eval_traits": eval_traits_for_train,
        "seeds": seeds,
        "alphas": alphas,
        "eval_option_rotations": eval_option_rotations,
        "eval_option_rotation_note": (
            "Only A-D substantive options are rotated. E remains the no-context fallback "
            "so the prompt contract's insufficient-context rule stays comparable."
        ),
        "train_per_class": args.train_per_class,
        "trait_specs": {
            name: {
                "attribute": spec.attribute,
                "positive_label": spec.positive_label,
                "negative_label": spec.negative_label,
                "positive_values": list(spec.positive_values),
                "negative_values": list(spec.negative_values),
                "probe_count": len(spec.probes),
            }
            for name, spec in TRAIT_SPECS.items()
        },
    }

    for seed in seeds:
        for train_trait_name in train_trait_names:
            spec = TRAIT_SPECS[train_trait_name]
            train_pos, train_neg, heldout_pos, heldout_neg = select_persona_groups(
                dataset,
                spec,
                train_per_class=args.train_per_class,
                seed=seed,
            )
            train_personas = train_pos + train_neg
            heldout_personas = heldout_pos + heldout_neg
            train_pos_ids = {persona.id for persona in train_pos}
            train_neg_ids = {persona.id for persona in train_neg}
            heldout_pos_ids = {persona.id for persona in heldout_pos}
            heldout_neg_ids = {persona.id for persona in heldout_neg}
            class_by_id = {
                **{persona.id: "positive" for persona in train_pos + heldout_pos},
                **{persona.id: "negative" for persona in train_neg + heldout_neg},
            }

            train_ids, train_masks, train_rows = prepare_probe_inputs(
                model,
                train_personas,
                spec,
                class_by_persona_id=class_by_id,
                activation_source=args.activation_source,
            )
            heldout_ids, heldout_masks, heldout_rows = prepare_probe_inputs(
                model,
                heldout_personas,
                spec,
                class_by_persona_id=class_by_id,
                activation_source=args.activation_source,
            )

            console.print(
                f"[cyan]seed={seed} trait={train_trait_name}: extracting layer {args.layer}; "
                f"train_prompts={len(train_ids)} heldout_prompts={len(heldout_ids)}[/]"
            )
            train_acts, train_rows, skipped_train = extract_probe_layer_activations(
                model,
                input_ids_list=train_ids,
                token_masks=train_masks,
                rows=train_rows,
                layer=args.layer,
                remote=args.remote,
                batch_size=args.extraction_batch_size,
                label=f"{train_trait_name} seed={seed} train",
            )
            heldout_acts, heldout_rows, skipped_heldout = extract_probe_layer_activations(
                model,
                input_ids_list=heldout_ids,
                token_masks=heldout_masks,
                rows=heldout_rows,
                layer=args.layer,
                remote=args.remote,
                batch_size=args.extraction_batch_size,
                label=f"{train_trait_name} seed={seed} heldout",
            )
            skipped_extractions.extend(
                [{**row, "seed": seed} for row in skipped_train + skipped_heldout]
            )

            trait_vector = make_vector(
                train_acts,
                train_rows,
                positive_ids=train_pos_ids,
                negative_ids=train_neg_ids,
            )
            rng = random.Random(seed + 17)
            train_ids_all = [persona.id for persona in train_personas]
            shuffled_positive = set(rng.sample(train_ids_all, k=len(train_pos_ids)))
            shuffled_negative = set(train_ids_all) - shuffled_positive
            control_vector = make_vector(
                train_acts,
                train_rows,
                positive_ids=shuffled_positive,
                negative_ids=shuffled_negative,
            )

            vector_key = f"{train_trait_name}__seed_{seed}"
            vector_payload[vector_key] = {
                "trait_vector": trait_vector.detach().cpu(),
                "shuffled_control_vector": control_vector.detach().cpu(),
                "trait": train_trait_name,
                "seed": seed,
                "layer": args.layer,
                "activation_source": args.activation_source,
            }

            projection.extend(
                {
                    **row,
                    "seed": seed,
                    "vector": "true_trait",
                }
                for row in projection_rows(
                    acts=train_acts,
                    rows=train_rows,
                    vector=trait_vector,
                    split="train",
                )
            )
            projection.extend(
                {
                    **row,
                    "seed": seed,
                    "vector": "true_trait",
                }
                for row in projection_rows(
                    acts=heldout_acts,
                    rows=heldout_rows,
                    vector=trait_vector,
                    split="heldout",
                )
            )

            for eval_trait_name in eval_traits_for_train[train_trait_name]:
                eval_spec = TRAIT_SPECS[eval_trait_name]
                bare_rows = run_with_remote_retry(
                    lambda eval_spec=eval_spec: score_probe_bank(
                        model,
                        spec=eval_spec,
                        reference_persona=reference_persona,
                        remote=args.remote,
                        option_rotations=eval_option_rotations,
                    ),
                    label=f"bare probes eval={eval_trait_name} seed={seed}",
                    retries=5,
                    sleep_seconds=10,
                )
                for row in bare_rows:
                    score_rows.append(
                        {
                            **row,
                            "train_trait": train_trait_name,
                            "seed": seed,
                            "condition": "bare",
                            "alpha": 0.0,
                            "activation_source": args.activation_source,
                        }
                    )

                for alpha in alphas:
                    for condition, vector, signed_alpha in (
                        ("true_positive", trait_vector, alpha),
                        ("true_negative", trait_vector, -alpha),
                        ("shuffled_positive", control_vector, alpha),
                    ):
                        rows = run_with_remote_retry(
                            lambda eval_spec=eval_spec, vector=vector, signed_alpha=signed_alpha: score_probe_bank(
                                model,
                                spec=eval_spec,
                                reference_persona=reference_persona,
                                remote=args.remote,
                                option_rotations=eval_option_rotations,
                                steering_layer=args.layer,
                                steering_vector=vector,
                                steering_alpha=signed_alpha,
                            ),
                            label=(
                                f"{condition} train={train_trait_name} eval={eval_trait_name} "
                                f"seed={seed} alpha={alpha}"
                            ),
                            retries=5,
                            sleep_seconds=10,
                        )
                        for row in rows:
                            score_rows.append(
                                {
                                    **row,
                                    "train_trait": train_trait_name,
                                    "seed": seed,
                                    "condition": condition,
                                    "alpha": alpha,
                                    "activation_source": args.activation_source,
                                }
                            )

    torch.save(vector_payload, out_dir / "vectors.pt")
    metadata = {
        **run_metadata,
        "skipped_extraction_count": len(skipped_extractions),
        "kept_score_rows": len(score_rows),
        "projection_rows": len(projection),
    }
    summary = {
        "score_summary": summarize_scores(score_rows),
        "projection_summary": summarize_projection(projection),
    }

    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (out_dir / "summary.json").write_text(json.dumps({"metadata": metadata, **summary}, indent=2))
    (out_dir / "skipped_extractions.json").write_text(json.dumps(skipped_extractions, indent=2))
    with (out_dir / "per_probe.jsonl").open("w") as handle:
        for row in score_rows:
            handle.write(json.dumps(row) + "\n")
    with (out_dir / "projection.jsonl").open("w") as handle:
        for row in projection:
            handle.write(json.dumps(row) + "\n")

    console.print_json(json.dumps(summary))
    console.print(f"[green]Wrote trait direction suite to {out_dir}[/]")


if __name__ == "__main__":
    main()
