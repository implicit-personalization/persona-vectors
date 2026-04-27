#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import torch
from dotenv import load_dotenv
from nnterp import StandardizedTransformer
from persona_data.environment import set_seed
from persona_data.prompts import format_messages
from persona_data.synth_persona import PersonaData, QAPair
from rich.console import Console

from persona_vectors.eval import choice_token_ids, score_choice_distribution_batched
from persona_vectors.mc_prompt_contract import (
    DEFAULT_UNSURE_OPTION,
    render_mc_generation_prompt,
)
from persona_vectors.steering_eval_utils import is_oom_error, run_with_remote_retry

console = Console()


FREE_RESPONSE_INSTRUCTION = (
    "You are answering an interview question as the person described below. "
    "Use the biography only as context for the person's lived perspective. "
    "Answer in first person, as a lightly cleaned interview transcript."
)


ATTRIBUTE_SPECS = {
    "political_views": {
        "positive_label": "liberal",
        "negative_label": "conservative",
        "positive_values": ("Liberal", "Slightly liberal"),
        "negative_values": (
            "Conservative",
            "Slightly conservative",
            "Extremely conservative",
        ),
        "keywords": (
            "politic",
            "government",
            "tax",
            "inequality",
            "institution",
            "tradition",
            "regulation",
            "public service",
            "safety net",
            "conservative",
            "liberal",
            "vote",
            "republican",
            "democrat",
            "constitution",
            "border",
            "authority",
            "justice",
            "policy",
        ),
    },
    "total_wealth": {
        "positive_label": "higher_wealth",
        "negative_label": "lower_wealth",
        "positive_values": (
            "$40,000 to $75,000",
            "$75,000 to $100,000",
            "$100,000 to $150,000",
            "$250,000 to $500,000",
        ),
        "negative_values": (
            "Less than $5,000",
            "$5,000 to $20,000",
            "$20,000 to $40,000",
        ),
        "keywords": (
            "money",
            "financial",
            "paycheck",
            "rent",
            "bill",
            "savings",
            "wealth",
            "afford",
            "cost",
            "debt",
            "cash",
            "income",
            "class",
            "poverty",
            "survival",
            "insurance",
            "work",
        ),
    },
}


@dataclass(frozen=True)
class PersonaRecord:
    id: str
    persona: dict
    templated_view: str
    biography_view: str

    @property
    def name(self) -> str:
        first = str(self.persona.get("first_name", "")).strip()
        last = str(self.persona.get("last_name", "")).strip()
        return " ".join(part for part in (first, last) if part) or self.id

    def to_persona_data(self) -> PersonaData:
        return PersonaData(
            id=self.id,
            persona=self.persona,
            templated_view=self.templated_view,
            biography_view=self.biography_view,
            statements_view="",
        )


@dataclass(frozen=True)
class FreeResponseRow:
    persona_id: str
    qid: str
    question: str
    answer: str
    tags: tuple[str, ...]
    axis_hint: str
    shared_harvest_hint: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "ARENA-style response-token mean direction suite for SynthPersona. "
            "It extracts masked means over existing Stage5 free-response answer "
            "tokens, then builds persona panel vectors and attribute binary vectors."
        )
    )
    parser.add_argument("--model", default="google/gemma-2-9b-it")
    parser.add_argument("--layer", type=int, default=41)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(
            "/Users/hengxuli/Repos/synth-persona/data/pilots/"
            "kimi_pipeline_v5_30persona_incremental_from_v4"
        ),
    )
    parser.add_argument(
        "--shared-mc-root",
        type=Path,
        default=Path(
            "/Users/hengxuli/Repos/synth-persona/data/releases/"
            "shared_mc_release_v1_30persona_56item"
        ),
    )
    parser.add_argument("--mode", choices=["persona", "attribute", "both"], default="both")
    parser.add_argument(
        "--attribute",
        choices=sorted(ATTRIBUTE_SPECS),
        default="political_views",
    )
    parser.add_argument(
        "--qa-filter",
        choices=["all", "attribute_keywords"],
        default="all",
        help=(
            "Free-response QA selector. Persona vectors usually use all QA; "
            "attribute vectors can use keyword-selected trait-relevant QA."
        ),
    )
    parser.add_argument(
        "--context-mode",
        choices=["none", "templated", "biography"],
        default="none",
        help=(
            "Context prepended before the free-response question during answer-span "
            "activation extraction. 'none' is closest to using the existing answer text "
            "as the response artifact and avoids long-biography OOMs."
        ),
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=0,
        help="Optional hard character cap for templated/biography extraction context.",
    )
    parser.add_argument("--qa-per-persona", type=int, default=6)
    parser.add_argument("--persona-limit", type=int, default=0)
    parser.add_argument("--mc-items-per-persona", type=int, default=12)
    parser.add_argument("--train-per-class", type=int, default=4)
    parser.add_argument("--seeds", default="1337")
    parser.add_argument("--alphas", default="0.25,0.5,1.0")
    parser.add_argument("--extraction-batch-size", type=int, default=2)
    parser.add_argument("--score-batch-size", type=int, default=6)
    parser.add_argument("--remote", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path("/Users/hengxuli/Repos/synth-persona/.env"),
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
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


def stable_seed(*parts: object) -> int:
    text = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def default_output_dir(*, model_name: str, layer: int, mode: str, qa_filter: str) -> Path:
    run_id = datetime.now().strftime("%Y%m%dT%H%M%SZ")
    model_dir = model_name.replace("/", "__")
    return (
        Path("artifacts")
        / "experiments"
        / "response_mean_direction_suite"
        / f"{run_id}__{model_dir}__layer_{layer}__{mode}__{qa_filter}"
    )


def load_personas(data_root: Path) -> dict[str, PersonaRecord]:
    path = data_root / "dataset_personas.jsonl"
    personas: dict[str, PersonaRecord] = {}
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            personas[row["id"]] = PersonaRecord(
                id=row["id"],
                persona=row["persona"],
                templated_view=row.get("templated_view", ""),
                biography_view=row.get("biography_view") or row.get("biography_md", ""),
            )
    if not personas:
        raise ValueError(f"No personas loaded from {path}")
    return personas


def load_free_response_rows(data_root: Path) -> dict[str, list[FreeResponseRow]]:
    path = data_root / "dataset_qa.jsonl"
    by_persona: dict[str, list[FreeResponseRow]] = defaultdict(list)
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("answer_format") != "free_text":
                continue
            by_persona[row["id"]].append(
                FreeResponseRow(
                    persona_id=row["id"],
                    qid=row["qid"],
                    question=row["question"],
                    answer=row["answer"],
                    tags=tuple(row.get("tags") or ()),
                    axis_hint=str(row.get("axis_hint") or ""),
                    shared_harvest_hint=str(row.get("shared_harvest_hint") or ""),
                )
            )
    if not by_persona:
        raise ValueError(f"No free-response QA rows loaded from {path}")
    return by_persona


def load_shared_mc_rows(shared_mc_root: Path) -> dict[str, list[QAPair]]:
    path = shared_mc_root / "dataset_qa.jsonl"
    by_persona: dict[str, list[QAPair]] = defaultdict(list)
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("answer_format") != "choice":
                continue
            choices = [str(choice) for choice in row["choices"]]
            if len(choices) == 4:
                choices = [*choices, DEFAULT_UNSURE_OPTION]
            by_persona[row["id"]].append(
                QAPair(
                    qid=row["qid"],
                    type=row.get("type", "implicit"),
                    question=row["question"],
                    answer=row["answer"],
                    difficulty=int(row.get("difficulty") or 3),
                    answer_format="choice",
                    choices=choices,
                    correct_choice_index=int(row["correct_choice_index"]),
                    evidence_sids=[],
                    tags=list(row.get("tags") or []),
                )
            )
    if not by_persona:
        raise ValueError(f"No shared MC rows loaded from {path}")
    return by_persona


def free_response_matches_attribute(row: FreeResponseRow, attribute: str) -> bool:
    keywords = ATTRIBUTE_SPECS[attribute]["keywords"]
    haystack = " ".join(
        [
            row.question,
            row.answer,
            " ".join(row.tags),
            row.axis_hint,
            row.shared_harvest_hint,
        ]
    ).lower()
    return any(keyword in haystack for keyword in keywords)


def select_free_response_rows(
    rows_by_persona: dict[str, list[FreeResponseRow]],
    *,
    persona_ids: list[str],
    qa_filter: str,
    attribute: str,
    qa_per_persona: int,
    seed: int,
) -> list[FreeResponseRow]:
    selected: list[FreeResponseRow] = []
    for persona_id in persona_ids:
        rows = list(rows_by_persona.get(persona_id, []))
        if qa_filter == "attribute_keywords":
            rows = [row for row in rows if free_response_matches_attribute(row, attribute)]
        elif qa_filter != "all":
            raise AssertionError(f"Unhandled qa_filter: {qa_filter}")
        rows.sort(key=lambda row: row.qid)
        rng = random.Random(stable_seed(seed, persona_id, qa_filter, attribute))
        rng.shuffle(rows)
        if qa_per_persona > 0:
            rows = rows[:qa_per_persona]
        selected.extend(rows)
    return selected


def render_free_response_with_answer(
    tokenizer,
    *,
    persona: PersonaRecord,
    qa: FreeResponseRow,
    context_mode: str,
    max_context_chars: int,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    context = ""
    context_label = ""
    if context_mode == "none":
        context = ""
    elif context_mode == "templated":
        context_label = "PERSON ATTRIBUTES"
        context = persona.templated_view.strip()
    elif context_mode == "biography":
        context_label = "PERSON BIOGRAPHY"
        context = persona.biography_view.strip()
    else:
        raise AssertionError(f"Unhandled context_mode: {context_mode}")
    if max_context_chars > 0 and len(context) > max_context_chars:
        context = context[:max_context_chars].rsplit(" ", maxsplit=1)[0].strip()

    context_block = f"\n\n{context_label}:\n{context}" if context else ""
    user_prompt = (
        f"{FREE_RESPONSE_INSTRUCTION}"
        f"{context_block}\n\n"
        "QUESTION:\n"
        f"{qa.question.strip()}\n\n"
        "Answer in first person."
    )
    messages = [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": qa.answer.strip()},
    ]
    full_prompt, answer_start = format_messages(messages, tokenizer)
    input_ids = tokenizer(
        full_prompt,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids[0]
    special_ids = set(int(token_id) for token_id in tokenizer.all_special_ids)
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for idx in range(answer_start, int(input_ids.shape[0])):
        if int(input_ids[idx]) not in special_ids:
            mask[idx] = True
    if not bool(mask.any()):
        raise ValueError(f"Empty answer mask for {persona.id} {qa.qid}")
    metadata = {
        "persona_id": persona.id,
        "persona_name": persona.name,
        "qid": qa.qid,
        "question": qa.question,
        "answer": qa.answer,
        "tags": list(qa.tags),
        "axis_hint": qa.axis_hint,
        "shared_harvest_hint": qa.shared_harvest_hint,
        "context_mode": context_mode,
        "context_char_count": len(context),
        "prompt_token_count": int(input_ids.shape[0]),
        "masked_token_count": int(mask.sum().item()),
    }
    return input_ids, mask, metadata


def build_extraction_inputs(
    model: StandardizedTransformer,
    *,
    personas: dict[str, PersonaRecord],
    free_rows: list[FreeResponseRow],
    context_mode: str,
    max_context_chars: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[dict]]:
    input_ids_list: list[torch.Tensor] = []
    token_masks: list[torch.Tensor] = []
    metadata_rows: list[dict] = []
    for row in free_rows:
        input_ids, token_mask, metadata = render_free_response_with_answer(
            model.tokenizer,
            persona=personas[row.persona_id],
            qa=row,
            context_mode=context_mode,
            max_context_chars=max_context_chars,
        )
        input_ids_list.append(input_ids)
        token_masks.append(token_mask)
        metadata_rows.append(metadata)
    return input_ids_list, token_masks, metadata_rows


def chunked_indices(n_items: int, chunk_size: int) -> Iterable[tuple[int, int]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    for start in range(0, n_items, chunk_size):
        yield start, min(start + chunk_size, n_items)


def pad_input_batch(
    input_ids_list: list[torch.Tensor],
    token_masks: list[torch.Tensor],
    *,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_len = max(int(input_ids.shape[0]) for input_ids in input_ids_list)
    input_ids = torch.full(
        (len(input_ids_list), max_len),
        fill_value=pad_token_id,
        dtype=torch.long,
    )
    attention_mask = torch.zeros((len(input_ids_list), max_len), dtype=torch.long)
    token_mask = torch.zeros((len(input_ids_list), max_len), dtype=torch.bool)
    for row_idx, (ids, mask) in enumerate(zip(input_ids_list, token_masks, strict=True)):
        length = int(ids.shape[0])
        input_ids[row_idx, :length] = ids
        attention_mask[row_idx, :length] = 1
        token_mask[row_idx, :length] = mask
    if not bool(token_mask.any()):
        raise ValueError("Padded token mask selected zero tokens")
    return input_ids, attention_mask, token_mask


def resolve_saved_tensor(tensor_like) -> torch.Tensor:
    if hasattr(tensor_like, "value") and getattr(tensor_like, "value") is not None:
        tensor_like = tensor_like.value
    if not isinstance(tensor_like, torch.Tensor):
        raise TypeError(f"Saved tensor did not resolve: {type(tensor_like)!r}")
    return tensor_like.detach().cpu()


def extract_layer_activations_batched(
    model: StandardizedTransformer,
    *,
    input_ids_list: list[torch.Tensor],
    token_masks: list[torch.Tensor],
    layer: int,
    remote: bool,
) -> torch.Tensor:
    pad_token_id = model.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = model.tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id")

    input_ids, attention_mask, token_mask = pad_input_batch(
        input_ids_list,
        token_masks,
        pad_token_id=int(pad_token_id),
    )
    with torch.no_grad(), model.trace(
        input_ids,
        attention_mask=attention_mask,
        remote=remote,
    ):
        mask_on_device = token_mask.to(device=model.layers_output[layer].device)
        hidden = model.layers_output[layer].float()
        counts = mask_on_device.sum(dim=1).clamp_min(1).unsqueeze(-1)
        means = (hidden * mask_on_device.unsqueeze(-1)).sum(dim=1) / counts
        saved = means.detach().cpu().save()
    return resolve_saved_tensor(saved)


def extract_with_adaptive_batches(
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

    def run_chunk(start: int, end: int) -> torch.Tensor:
        return extract_layer_activations_batched(
            model,
            input_ids_list=input_ids_list[start:end],
            token_masks=token_masks[start:end],
            layer=layer,
            remote=remote,
        )

    for start, end in chunked_indices(len(input_ids_list), batch_size):
        try:
            chunk = run_with_remote_retry(
                lambda start=start, end=end: run_chunk(start, end),
                label=f"{label} extraction {start + 1}-{end}",
                retries=1,
                sleep_seconds=5,
            )
            chunks.append(chunk)
            kept_rows.extend(rows[start:end])
            continue
        except Exception as exc:
            if end - start == 1:
                if is_oom_error(exc):
                    skipped_rows.append(
                        {
                            **rows[start],
                            "skip_reason": "remote_oom",
                            "error_type": type(exc).__name__,
                            "error": str(exc)[:500],
                        }
                    )
                    console.print(
                        f"[yellow]Skipping {label} row {start + 1} after remote OOM: "
                        f"{rows[start]['persona_id']} {rows[start]['qid']}[/]"
                    )
                    continue
                raise
            console.print(
                f"[yellow]{label} batch {start + 1}-{end} failed; "
                f"retrying as single examples[/]"
            )

        for single in range(start, end):
            try:
                chunk = run_with_remote_retry(
                    lambda single=single: run_chunk(single, single + 1),
                    label=f"{label} extraction {single + 1}",
                    retries=2,
                    sleep_seconds=5,
                )
            except Exception as exc:
                if is_oom_error(exc):
                    skipped_rows.append(
                        {
                            **rows[single],
                            "skip_reason": "remote_oom",
                            "error_type": type(exc).__name__,
                            "error": str(exc)[:500],
                        }
                    )
                    continue
                raise
            chunks.append(chunk)
            kept_rows.append(rows[single])

    if not chunks:
        raise RuntimeError(f"No activations extracted for {label}")
    return torch.cat(chunks, dim=0), kept_rows, skipped_rows


def mean_for_personas(
    acts: torch.Tensor,
    rows: list[dict],
    persona_ids: set[str],
) -> torch.Tensor:
    indices = [idx for idx, row in enumerate(rows) if row["persona_id"] in persona_ids]
    if not indices:
        raise ValueError(f"No rows for persona ids: {sorted(persona_ids)}")
    return acts[indices].float().mean(dim=0)


def build_persona_vectors(
    acts: torch.Tensor,
    rows: list[dict],
    persona_ids: list[str],
) -> dict[str, torch.Tensor]:
    vectors: dict[str, torch.Tensor] = {}
    all_ids = set(persona_ids)
    for persona_id in persona_ids:
        own = mean_for_personas(acts, rows, {persona_id})
        others = mean_for_personas(acts, rows, all_ids - {persona_id})
        vectors[persona_id] = own - others
    return vectors


def select_attribute_groups(
    personas: dict[str, PersonaRecord],
    *,
    attribute: str,
    train_per_class: int,
    seed: int,
    eligible_persona_ids: set[str],
) -> tuple[list[str], list[str], list[str], list[str]]:
    spec = ATTRIBUTE_SPECS[attribute]
    positive_values = set(spec["positive_values"])
    negative_values = set(spec["negative_values"])
    positives = [
        persona_id
        for persona_id, persona in personas.items()
        if persona_id in eligible_persona_ids
        if str(persona.persona.get(attribute, "")).strip() in positive_values
    ]
    negatives = [
        persona_id
        for persona_id, persona in personas.items()
        if persona_id in eligible_persona_ids
        if str(persona.persona.get(attribute, "")).strip() in negative_values
    ]
    rng = random.Random(seed)
    positives.sort()
    negatives.sort()
    rng.shuffle(positives)
    rng.shuffle(negatives)
    if len(positives) < train_per_class + 1 or len(negatives) < train_per_class + 1:
        raise ValueError(
            f"{attribute} needs at least train_per_class+1 examples per class; "
            f"got positives={len(positives)} negatives={len(negatives)}"
        )
    return (
        positives[:train_per_class],
        negatives[:train_per_class],
        positives[train_per_class:],
        negatives[train_per_class:],
    )


def build_attribute_vector(
    acts: torch.Tensor,
    rows: list[dict],
    *,
    positive_ids: set[str],
    negative_ids: set[str],
) -> torch.Tensor:
    return mean_for_personas(acts, rows, positive_ids) - mean_for_personas(
        acts,
        rows,
        negative_ids,
    )


def projection_rows(
    *,
    acts: torch.Tensor,
    rows: list[dict],
    vector: torch.Tensor,
    split: str,
    class_by_persona_id: dict[str, str],
    seed: int,
    vector_name: str,
) -> list[dict]:
    unit = vector.float() / (vector.float().norm() + 1e-8)
    output: list[dict] = []
    for idx, row in enumerate(rows):
        label = class_by_persona_id.get(row["persona_id"])
        if label is None:
            continue
        output.append(
            {
                **row,
                "split": split,
                "class_label": label,
                "seed": seed,
                "vector": vector_name,
                "projection": float(acts[idx].float().dot(unit).item()),
            }
        )
    return output


def summarize_projection(rows: list[dict]) -> dict:
    grouped: dict[tuple[str, str, int, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["split"], row["class_label"], row["seed"], row["vector"])].append(
            float(row["projection"])
        )
    summary = {
        f"split={split}::class={class_label}::seed={seed}::vector={vector}": {
            "n": len(values),
            "mean_projection": sum(values) / len(values),
        }
        for (split, class_label, seed, vector), values in sorted(grouped.items())
    }
    for split, seed, vector in sorted(
        {(split, seed, vector) for split, _, seed, vector in grouped}
    ):
        positive = grouped.get((split, "positive", seed, vector), [])
        negative = grouped.get((split, "negative", seed, vector), [])
        if not positive or not negative:
            continue
        wins = 0.0
        for pos_value in positive:
            for neg_value in negative:
                if pos_value > neg_value:
                    wins += 1.0
                elif pos_value == neg_value:
                    wins += 0.5
        summary[
            f"split={split}::seed={seed}::vector={vector}::positive_vs_negative"
        ] = {
            "positive_n": len(positive),
            "negative_n": len(negative),
            "positive_minus_negative_mean_projection": (
                sum(positive) / len(positive) - sum(negative) / len(negative)
            ),
            "pairwise_auc_positive_greater_than_negative": (
                wins / (len(positive) * len(negative))
            ),
        }
    return summary


def select_mc_rows(
    mc_by_persona: dict[str, list[QAPair]],
    *,
    persona_id: str,
    limit: int,
    seed: int,
) -> list[QAPair]:
    rows = list(mc_by_persona.get(persona_id, []))
    rows.sort(key=lambda qa: qa.qid)
    rng = random.Random(stable_seed(seed, persona_id, "shared_mc"))
    rng.shuffle(rows)
    if limit > 0:
        rows = rows[:limit]
    return rows


def score_shared_mc_for_persona_vectors(
    model: StandardizedTransformer,
    *,
    personas: dict[str, PersonaRecord],
    mc_by_persona: dict[str, list[QAPair]],
    persona_vectors: dict[str, torch.Tensor],
    persona_ids: list[str],
    layer: int,
    alphas: list[float],
    remote: bool,
    score_batch_size: int,
    mc_items_per_persona: int,
    seed: int,
) -> list[dict]:
    rows: list[dict] = []
    prompts: list[str] = []
    prompt_lens: list[int] = []
    choice_ids_list: list[list[int]] = []
    eval_meta: list[dict] = []

    for persona_id in persona_ids:
        persona = personas[persona_id]
        persona_data = persona.to_persona_data()
        for qa in select_mc_rows(
            mc_by_persona,
            persona_id=persona_id,
            limit=mc_items_per_persona,
            seed=seed,
        ):
            prompt, prompt_len = render_mc_generation_prompt(
                model.tokenizer,
                persona=persona_data,
                qa=qa,
                condition="bare",
            )
            _, choice_ids = choice_token_ids(model.tokenizer, qa)
            prompts.append(prompt)
            prompt_lens.append(prompt_len)
            choice_ids_list.append(choice_ids)
            eval_meta.append(
                {
                    "persona_id": persona_id,
                    "persona_name": persona.name,
                    "qid": qa.qid,
                    "question": qa.question,
                    "gold_index": qa.correct_choice_index,
                    "gold_letter": chr(ord("A") + int(qa.correct_choice_index)),
                    "choices": qa.choices,
                }
            )
    other_vector_by_persona: dict[str, str] = {}
    for persona_id in persona_ids:
        candidates = [candidate for candidate in persona_ids if candidate != persona_id]
        if not candidates:
            continue
        rng = random.Random(stable_seed(seed, persona_id, "random_other_vector"))
        other_vector_by_persona[persona_id] = rng.choice(candidates)

    def append_score_rows(
        *,
        condition: str,
        alpha: float | None,
        logprobs: list[torch.Tensor],
        probs: list[torch.Tensor],
        source_vector_persona_id: str | None,
    ) -> None:
        for meta, lp, prob in zip(eval_meta, logprobs, probs, strict=True):
            gold_idx = int(meta["gold_index"])
            pred_idx = int(lp.argmax().item())
            other_idxs = [idx for idx in range(len(meta["choices"])) if idx != gold_idx]
            best_other = float(lp[other_idxs].max().item()) if other_idxs else float("-inf")
            rows.append(
                {
                    **meta,
                    "condition": condition,
                    "alpha": alpha,
                    "source_vector_persona_id": source_vector_persona_id,
                    "predicted_letter": chr(ord("A") + pred_idx),
                    "correct": pred_idx == gold_idx,
                    "gold_prob": float(prob[gold_idx].item()),
                    "gold_logprob": float(lp[gold_idx].item()),
                    "margin_vs_best_other": float(lp[gold_idx].item() - best_other),
                    "choice_probs": [float(value) for value in prob.tolist()],
                    "choice_logprobs": [float(value) for value in lp.tolist()],
                }
            )

    bare_logprobs: list[torch.Tensor] = []
    bare_probs: list[torch.Tensor] = []
    for start, end in chunked_indices(len(prompts), score_batch_size):
        lp, prob = score_choice_distribution_batched(
            model,
            prompts[start:end],
            prompt_lens[start:end],
            choice_ids_list[start:end],
            remote=remote,
        )
        bare_logprobs.extend(lp)
        bare_probs.extend(prob)
    append_score_rows(
        condition="bare",
        alpha=None,
        logprobs=bare_logprobs,
        probs=bare_probs,
        source_vector_persona_id=None,
    )

    for alpha in alphas:
        for persona_id in persona_ids:
            target_indices = [
                idx for idx, meta in enumerate(eval_meta) if meta["persona_id"] == persona_id
            ]
            if not target_indices:
                continue
            target_prompts = [prompts[idx] for idx in target_indices]
            target_lens = [prompt_lens[idx] for idx in target_indices]
            target_choices = [choice_ids_list[idx] for idx in target_indices]
            vector_conditions = [("own_response_mean_vector", persona_id)]
            if persona_id in other_vector_by_persona:
                vector_conditions.append(
                    (
                        "random_other_response_mean_vector",
                        other_vector_by_persona[persona_id],
                    )
                )
            for condition, source_persona_id in vector_conditions:
                vector = persona_vectors[source_persona_id]
                steered_logprobs: list[torch.Tensor] = []
                steered_probs: list[torch.Tensor] = []
                for start, end in chunked_indices(len(target_prompts), score_batch_size):
                    lp, prob = score_choice_distribution_batched(
                        model,
                        target_prompts[start:end],
                        target_lens[start:end],
                        target_choices[start:end],
                        remote=remote,
                        steering_layer=layer,
                        steering_vector=vector,
                        steering_alpha=alpha,
                    )
                    steered_logprobs.extend(lp)
                    steered_probs.extend(prob)
                for local_idx, lp, prob in zip(
                    target_indices,
                    steered_logprobs,
                    steered_probs,
                    strict=True,
                ):
                    meta = eval_meta[local_idx]
                    gold_idx = int(meta["gold_index"])
                    pred_idx = int(lp.argmax().item())
                    other_idxs = [
                        idx for idx in range(len(meta["choices"])) if idx != gold_idx
                    ]
                    best_other = (
                        float(lp[other_idxs].max().item()) if other_idxs else float("-inf")
                    )
                    rows.append(
                        {
                            **meta,
                            "condition": condition,
                            "alpha": alpha,
                            "source_vector_persona_id": source_persona_id,
                            "predicted_letter": chr(ord("A") + pred_idx),
                            "correct": pred_idx == gold_idx,
                            "gold_prob": float(prob[gold_idx].item()),
                            "gold_logprob": float(lp[gold_idx].item()),
                            "margin_vs_best_other": float(lp[gold_idx].item() - best_other),
                            "choice_probs": [float(value) for value in prob.tolist()],
                            "choice_logprobs": [float(value) for value in lp.tolist()],
                        }
                    )

    return rows


def summarize_mc_scores(rows: list[dict]) -> dict:
    grouped: dict[tuple[str, float | None], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["condition"], row["alpha"])].append(row)

    bare_by_key = {
        (row["persona_id"], row["qid"]): row
        for row in rows
        if row["condition"] == "bare"
    }
    summary: dict[str, dict] = {}
    for (condition, alpha), condition_rows in sorted(
        grouped.items(),
        key=lambda item: (item[0][0], -1 if item[0][1] is None else item[0][1]),
    ):
        key = f"condition={condition}::alpha={alpha}"
        n_rows = len(condition_rows)
        mean_gold_logprob = sum(row["gold_logprob"] for row in condition_rows) / n_rows
        mean_gold_prob = sum(row["gold_prob"] for row in condition_rows) / n_rows
        mean_margin = sum(row["margin_vs_best_other"] for row in condition_rows) / n_rows
        payload = {
            "n": n_rows,
            "accuracy": sum(int(row["correct"]) for row in condition_rows) / n_rows,
            "mean_gold_prob": mean_gold_prob,
            "mean_gold_logprob": mean_gold_logprob,
            "mean_margin_vs_best_other": mean_margin,
            "predicted_letter_counts": dict(
                Counter(row["predicted_letter"] for row in condition_rows)
            ),
        }
        if condition != "bare":
            comparable = []
            for row in condition_rows:
                bare = bare_by_key.get((row["persona_id"], row["qid"]))
                if bare is not None:
                    comparable.append((row, bare))
            if comparable:
                payload["delta_vs_bare_gold_logprob"] = sum(
                    row["gold_logprob"] - bare["gold_logprob"]
                    for row, bare in comparable
                ) / len(comparable)
                payload["delta_vs_bare_gold_prob"] = sum(
                    row["gold_prob"] - bare["gold_prob"] for row, bare in comparable
                ) / len(comparable)
                payload["changed_predicted_letter_rate"] = sum(
                    int(row["predicted_letter"] != bare["predicted_letter"])
                    for row, bare in comparable
                ) / len(comparable)
        summary[key] = payload
    return summary


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def main() -> None:
    args = parse_args()
    load_dotenv()
    if args.env_file is not None:
        load_dotenv(args.env_file, override=False)
    if args.remote and not args.dry_run and not os.environ.get("NDIF_API_KEY"):
        raise RuntimeError("NDIF_API_KEY is required for remote runs.")

    seeds = parse_int_csv(args.seeds)
    alphas = parse_float_csv(args.alphas)
    out_dir = args.out_dir or default_output_dir(
        model_name=args.model,
        layer=args.layer,
        mode=args.mode,
        qa_filter=args.qa_filter,
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    personas = load_personas(args.data_root)
    free_by_persona = load_free_response_rows(args.data_root)
    mc_by_persona = load_shared_mc_rows(args.shared_mc_root)
    persona_ids = sorted(personas)
    if args.persona_limit > 0:
        persona_ids = persona_ids[: args.persona_limit]

    selected_free_rows = select_free_response_rows(
        free_by_persona,
        persona_ids=persona_ids,
        qa_filter=args.qa_filter,
        attribute=args.attribute,
        qa_per_persona=args.qa_per_persona,
        seed=seeds[0],
    )
    selected_counts = Counter(row.persona_id for row in selected_free_rows)
    metadata = {
        "preflight": {
            "target_behavior": (
                "Move no-context shared MC answer probabilities toward the target "
                "persona's biography-first gold choices, and check whether attribute "
                "labels are linearly separated in free-response answer activations."
            ),
            "internal_object": (
                f"Layer {args.layer} mean residual activation over existing Stage5 "
                "assistant answer tokens, using the configured context mode plus "
                "question + answer chat format."
            ),
            "claim": (
                "Response-token mean directions are a better SynthPersona vector source "
                "than MC prompt-last or forced answer-letter directions."
            ),
            "mechanistic_need": (
                "Pure MC accuracy only tests prompting; this run tests whether an "
                "activation direction extracted from free-response answer spans transfers "
                "to held-out MC answer-time logits."
            ),
            "smallest_falsification": (
                "Persona vectors do not improve shared MC gold logprob versus bare, and "
                "attribute vectors fail heldout projection separation."
            ),
            "redesign_failure_case": (
                "Projection separates labels but ActAdd still does not move MC logits, "
                "which would mean this is a readout direction rather than a useful steering direction."
            ),
        },
        "model": args.model,
        "layer": args.layer,
        "mode": args.mode,
        "attribute": args.attribute,
        "data_root": str(args.data_root),
        "shared_mc_root": str(args.shared_mc_root),
        "qa_filter": args.qa_filter,
        "context_mode": args.context_mode,
        "max_context_chars": args.max_context_chars,
        "qa_per_persona": args.qa_per_persona,
        "persona_limit": args.persona_limit,
        "selected_personas": len(persona_ids),
        "selected_free_response_rows": len(selected_free_rows),
        "min_free_response_rows_per_persona": min(selected_counts.values()),
        "max_free_response_rows_per_persona": max(selected_counts.values()),
        "mc_items_per_persona": args.mc_items_per_persona,
        "train_per_class": args.train_per_class,
        "seeds": seeds,
        "alphas": alphas,
        "extraction_batch_size": args.extraction_batch_size,
        "score_batch_size": args.score_batch_size,
        "remote": args.remote,
        "dry_run": args.dry_run,
        "attribute_spec": {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in ATTRIBUTE_SPECS[args.attribute].items()
        },
    }

    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    write_jsonl(
        out_dir / "selected_free_response_rows.jsonl",
        [
            {
                "persona_id": row.persona_id,
                "qid": row.qid,
                "question": row.question,
                "answer": row.answer,
                "tags": list(row.tags),
                "axis_hint": row.axis_hint,
                "shared_harvest_hint": row.shared_harvest_hint,
            }
            for row in selected_free_rows
        ],
    )

    if args.dry_run:
        summary = {
            "metadata": metadata,
            "selected_free_response_rows_by_persona": dict(selected_counts),
            "shared_mc_rows_by_persona": {
                persona_id: len(mc_by_persona.get(persona_id, []))
                for persona_id in persona_ids
            },
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        console.print_json(json.dumps(summary))
        console.print(f"[green]Dry run wrote response-mean plan to {out_dir}[/]")
        return

    set_seed(seeds[0])
    model = StandardizedTransformer(args.model)
    input_ids, token_masks, extraction_rows = build_extraction_inputs(
        model,
        personas=personas,
        free_rows=selected_free_rows,
        context_mode=args.context_mode,
        max_context_chars=args.max_context_chars,
    )
    acts, kept_rows, skipped_rows = extract_with_adaptive_batches(
        model,
        input_ids_list=input_ids,
        token_masks=token_masks,
        rows=extraction_rows,
        layer=args.layer,
        remote=args.remote,
        batch_size=args.extraction_batch_size,
        label="response_mean",
    )
    write_jsonl(out_dir / "extraction_rows.jsonl", kept_rows)
    (out_dir / "skipped_extractions.json").write_text(
        json.dumps(skipped_rows, indent=2)
    )

    vector_payload: dict[str, torch.Tensor] = {}
    projection: list[dict] = []
    mc_rows: list[dict] = []
    persona_vectors: dict[str, torch.Tensor] = {}

    if args.mode in ("persona", "both"):
        persona_vectors = build_persona_vectors(acts, kept_rows, persona_ids)
        for persona_id, vector in persona_vectors.items():
            vector_payload[f"persona::{persona_id}"] = vector.detach().cpu()
        mc_rows = score_shared_mc_for_persona_vectors(
            model,
            personas=personas,
            mc_by_persona=mc_by_persona,
            persona_vectors=persona_vectors,
            persona_ids=persona_ids,
            layer=args.layer,
            alphas=alphas,
            remote=args.remote,
            score_batch_size=args.score_batch_size,
            mc_items_per_persona=args.mc_items_per_persona,
            seed=seeds[0],
        )
        write_jsonl(out_dir / "shared_mc_scores.jsonl", mc_rows)

    if args.mode in ("attribute", "both"):
        for seed in seeds:
            train_pos, train_neg, heldout_pos, heldout_neg = select_attribute_groups(
                personas,
                attribute=args.attribute,
                train_per_class=args.train_per_class,
                seed=seed,
                eligible_persona_ids=set(persona_ids),
            )
            class_by_persona_id = {
                **{persona_id: "positive" for persona_id in train_pos + heldout_pos},
                **{persona_id: "negative" for persona_id in train_neg + heldout_neg},
            }
            true_vector = build_attribute_vector(
                acts,
                kept_rows,
                positive_ids=set(train_pos),
                negative_ids=set(train_neg),
            )
            rng = random.Random(seed + 17)
            train_ids = train_pos + train_neg
            shuffled_pos = set(rng.sample(train_ids, k=len(train_pos)))
            shuffled_neg = set(train_ids) - shuffled_pos
            shuffled_vector = build_attribute_vector(
                acts,
                kept_rows,
                positive_ids=shuffled_pos,
                negative_ids=shuffled_neg,
            )
            vector_payload[f"attribute::{args.attribute}::seed_{seed}"] = (
                true_vector.detach().cpu()
            )
            vector_payload[f"attribute_control::{args.attribute}::seed_{seed}"] = (
                shuffled_vector.detach().cpu()
            )

            train_ids_set = set(train_pos + train_neg)
            heldout_ids_set = set(heldout_pos + heldout_neg)
            train_indices = [
                idx for idx, row in enumerate(kept_rows) if row["persona_id"] in train_ids_set
            ]
            heldout_indices = [
                idx for idx, row in enumerate(kept_rows) if row["persona_id"] in heldout_ids_set
            ]
            train_acts = acts[train_indices]
            train_rows = [kept_rows[idx] for idx in train_indices]
            heldout_acts = acts[heldout_indices]
            heldout_rows = [kept_rows[idx] for idx in heldout_indices]
            projection.extend(
                projection_rows(
                    acts=train_acts,
                    rows=train_rows,
                    vector=true_vector,
                    split="train",
                    class_by_persona_id=class_by_persona_id,
                    seed=seed,
                    vector_name="true_attribute",
                )
            )
            projection.extend(
                projection_rows(
                    acts=heldout_acts,
                    rows=heldout_rows,
                    vector=true_vector,
                    split="heldout",
                    class_by_persona_id=class_by_persona_id,
                    seed=seed,
                    vector_name="true_attribute",
                )
            )
            projection.extend(
                projection_rows(
                    acts=train_acts,
                    rows=train_rows,
                    vector=shuffled_vector,
                    split="train",
                    class_by_persona_id=class_by_persona_id,
                    seed=seed,
                    vector_name="shuffled_control",
                )
            )
            projection.extend(
                projection_rows(
                    acts=heldout_acts,
                    rows=heldout_rows,
                    vector=shuffled_vector,
                    split="heldout",
                    class_by_persona_id=class_by_persona_id,
                    seed=seed,
                    vector_name="shuffled_control",
                )
            )
        write_jsonl(out_dir / "attribute_projection.jsonl", projection)

    torch.save(vector_payload, out_dir / "vectors.pt")

    summary = {
        "metadata": {
            **metadata,
            "kept_extractions": len(kept_rows),
            "skipped_extractions": len(skipped_rows),
            "vector_count": len(vector_payload),
            "shared_mc_score_rows": len(mc_rows),
            "attribute_projection_rows": len(projection),
        },
        "persona_shared_mc_summary": summarize_mc_scores(mc_rows) if mc_rows else {},
        "attribute_projection_summary": summarize_projection(projection)
        if projection
        else {},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    console.print_json(json.dumps(summary))
    console.print(f"[green]Wrote response-mean direction suite to {out_dir}[/]")


if __name__ == "__main__":
    main()
