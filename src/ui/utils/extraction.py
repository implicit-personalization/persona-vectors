from dataclasses import dataclass

import torch

from src.activation_io import save_per_question_vectors
from src.activations import extract_activations
from src.environment import get_artifacts_dir
from src.prompt_format import (
    format_biography_prompt,
    format_messages,
    format_templated_prompt,
)
from src.synth_persona_io import PersonaData, QAPair


@dataclass
class VariantExtractionResult:
    variant: str
    output_dir: str
    n_questions: int
    n_layers: int
    d_model: int


def _prepare_inputs(
    tokenizer,
    system_prompt: str,
    qa_pairs: list[QAPair],
) -> tuple[list[str], list[torch.Tensor], list[str]]:
    full_texts: list[str] = []
    token_masks: list[torch.Tensor] = []
    questions: list[str] = []

    for qa in qa_pairs:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": qa.question},
            {"role": "assistant", "content": qa.answer},
        ]
        full_prompt, answer_start = format_messages(messages, tokenizer)
        seq_len = tokenizer(full_prompt, return_tensors="pt").input_ids.shape[1]

        full_texts.append(full_prompt)
        token_masks.append(torch.arange(seq_len) >= answer_start)
        questions.append(qa.question)

    return full_texts, token_masks, questions


def run_extraction(
    model,
    model_name: str,
    persona: PersonaData,
    qa_pairs: list[QAPair],
    variants: list[str],
    remote: bool,
) -> list[VariantExtractionResult]:
    """Run activation extraction and save outputs for selected variants."""
    if not qa_pairs:
        raise ValueError("No QA pairs selected for extraction")

    tokenizer = model.tokenizer
    activations_dir = get_artifacts_dir() / "activations"

    system_prompt_by_variant = {
        "templated": format_templated_prompt(persona.templated_prompt),
        "biography": format_biography_prompt(persona.biography_md),
    }

    results: list[VariantExtractionResult] = []

    for variant in variants:
        if variant not in system_prompt_by_variant:
            raise ValueError(f"Unsupported variant: {variant}")

        full_texts, token_masks, questions = _prepare_inputs(
            tokenizer=tokenizer,
            system_prompt=system_prompt_by_variant[variant],
            qa_pairs=qa_pairs,
        )

        per_question_vectors = extract_activations(
            model=model,
            full_texts=full_texts,
            token_masks=token_masks,
            remote=remote,
        )

        artifact_dir = save_per_question_vectors(
            root_dir=activations_dir,
            model_name=model_name,
            prompt_variant=variant,
            persona_id=persona.id,
            persona_name=persona.name,
            per_question_vectors=per_question_vectors,
            questions=questions,
        )

        results.append(
            VariantExtractionResult(
                variant=variant,
                output_dir=str(artifact_dir),
                n_questions=per_question_vectors.shape[0],
                n_layers=per_question_vectors.shape[1],
                d_model=per_question_vectors.shape[2],
            )
        )

    return results
