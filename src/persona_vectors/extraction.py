import gc
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from nnterp import StandardizedTransformer
from persona_data.environment import get_artifacts_dir
from persona_data.prompts import (
    format_biography_prompt,
    format_messages,
    format_templated_prompt,
)
from persona_data.synth_persona import PersonaData, QAPair

from persona_vectors.activation_io import save_per_question_vectors
from persona_vectors.activations import extract_activations

logger = logging.getLogger(__name__)

_VARIANT_PROMPTS: dict[str, Callable[[PersonaData], str]] = {
    "templated": lambda p: format_templated_prompt(p.templated_prompt),
    "biography": lambda p: format_biography_prompt(p.biography_md),
}

SUPPORTED_VARIANTS: tuple[str, ...] = tuple(_VARIANT_PROMPTS)


@dataclass
class ExtractionResult:
    variant: str
    output_dir: Path
    n_questions: int
    persona_name: str


def _prepare_inputs(
    tokenizer,
    system_prompt: str,
    qa_pairs: list[QAPair],
) -> tuple[list[str], list[torch.Tensor], list[str]]:
    full_texts, token_masks, questions = [], [], []
    for qa in qa_pairs:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": qa.question},
            {"role": "assistant", "content": qa.answer},
        ]
        full_prompt, answer_start = format_messages(messages, tokenizer)
        seq_len = tokenizer(full_prompt, return_tensors="pt").input_ids.shape[1]
        full_texts.append(full_prompt)
        # NOTE: Get average over all response tokens
        # This is flexible and can be change accordingly
        token_masks.append(torch.arange(seq_len) >= answer_start)
        questions.append(qa.question)
    return full_texts, token_masks, questions


def run_extraction(
    model: StandardizedTransformer,
    model_name: str,
    persona: PersonaData,
    qa_pairs: list[QAPair],
    variants: list[str],
    remote: bool = False,
) -> list[ExtractionResult]:
    """Extract and save per-question activation vectors for each prompt variant.

    Args:
        model: Loaded standardized nnterp model.
        model_name: HuggingFace model identifier used for artifact paths.
        persona: The persona whose QA pairs are being extracted.
        qa_pairs: Question-answer pairs to run extraction on.
        variants: Prompt variants to extract (``"templated"`` or ``"biography"``).
        remote: Whether to execute on NDIF.

    Returns:
        One ExtractionResult per variant.
    """
    if not qa_pairs:
        raise ValueError("No QA pairs selected for extraction")
    if invalid := set(variants) - set(SUPPORTED_VARIANTS):
        raise ValueError(f"Unsupported variants: {invalid}")

    activations_dir = get_artifacts_dir() / "activations"
    results = []

    for variant in variants:
        full_texts, token_masks, questions = _prepare_inputs(
            tokenizer=model.tokenizer,
            system_prompt=_VARIANT_PROMPTS[variant](persona),
            qa_pairs=qa_pairs,
        )

        vectors = extract_activations(model, full_texts, token_masks, remote=remote)

        artifact_dir = save_per_question_vectors(
            root_dir=activations_dir,
            model_name=model_name,
            prompt_variant=variant,
            persona_id=persona.id,
            persona_name=persona.name,
            per_question_vectors=vectors,
            questions=questions,
        )

        results.append(
            ExtractionResult(
                variant=variant,
                output_dir=artifact_dir,
                n_questions=vectors.shape[0],
                persona_name=persona.name,
            )
        )

        # Free tensors between variants to keep memory bounded.
        del vectors, full_texts, token_masks
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

    return results
