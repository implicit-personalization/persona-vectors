import gc
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Callable

import torch
from nnterp import StandardizedTransformer
from persona_data.prompts import (
    format_mc_question,
    format_messages,
    format_roleplay_prompt,
    mc_correct_letter,
)
from persona_data.synth_persona import PersonaData, QAPair
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from persona_vectors.activations import extract_activations
from persona_vectors.artifacts import SUPPORTED_VARIANTS, ActivationStore

_MASK_STYLE = "black on green"
_PROMPT_STYLE = "dim"
_RESPONSE_STYLE = "bright_cyan"
_SPECIAL_STYLE = "bold magenta"


class MaskStrategy(StrEnum):
    """Which tokens contribute to the averaged hidden state.

    All strategies pivot around the assistant response span, anchored at
    ``answer_start`` (the first response token in the tokenized full prompt).
    ``prompt_last`` is the token immediately before ``answer_start``, which is
    often a newline or chat-template delimiter rather than a visible word.
    ``prompt_last_special`` walks backwards from ``answer_start`` to find the
    last special token on the prompt side (e.g. ``<end_of_turn>``), which may
    act as a summary position in the residual stream.
    """

    RESPONSE_MEAN = "response_mean"
    RESPONSE_FIRST = "response_first"
    RESPONSE_LAST = "response_last"
    PROMPT_MEAN = "prompt_mean"
    PROMPT_LAST = "prompt_last"
    PROMPT_LAST_SPECIAL = "prompt_last_special"


@dataclass
class ExtractionResult:
    variant: str
    output_dir: Path
    n_questions: int
    persona_name: str


@dataclass
class PreparedInput:
    """A single formatted sample ready for activation extraction.

    Attributes:
        question: Original user question text (used for artifact metadata).
        input_ids: Token ids for the formatted prompt (shape ``(seq_len,)``).
        answer_start: Index of the first assistant token in ``input_ids``.
        token_mask: Boolean mask over ``input_ids`` selecting which tokens
            contribute to the averaged hidden state.
    """

    question: str
    input_ids: torch.Tensor
    answer_start: int
    token_mask: torch.Tensor


def _build_messages(qa: QAPair, system_prompt: str) -> list[dict[str, str]]:
    if qa.answer_format == "choice" and qa.correct_choice_index is not None:
        user_content = format_mc_question(qa)
        answer_content = mc_correct_letter(qa)
    else:
        user_content = qa.question
        answer_content = qa.answer
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": answer_content},
    ]


def _build_mask(
    seq_len: int,
    answer_start: int,
    answer_end: int,
    strategy: MaskStrategy,
    input_ids: torch.Tensor | None = None,
    special_ids: set[int] | None = None,
) -> torch.Tensor:
    """Return a boolean mask over ``seq_len`` tokens for the given strategy."""
    if answer_start <= 0 or answer_start >= seq_len:
        raise ValueError(f"Invalid answer_start={answer_start} for seq_len={seq_len}")
    if answer_end <= answer_start or answer_end > seq_len:
        raise ValueError(
            f"Invalid answer_end={answer_end} for answer_start={answer_start} and seq_len={seq_len}"
        )

    mask = torch.zeros(seq_len, dtype=torch.bool)
    if strategy is MaskStrategy.RESPONSE_MEAN:
        mask[answer_start:answer_end] = True
    elif strategy is MaskStrategy.RESPONSE_FIRST:
        mask[answer_start] = True
    elif strategy is MaskStrategy.RESPONSE_LAST:
        mask[answer_end - 1] = True
    elif strategy is MaskStrategy.PROMPT_MEAN:
        mask[:answer_start] = True
    elif strategy is MaskStrategy.PROMPT_LAST:
        # This is the last prompt-side token before the assistant response.
        mask[answer_start - 1] = True
    elif strategy is MaskStrategy.PROMPT_LAST_SPECIAL:
        # Last special/delimiter token on the prompt side (e.g. <end_of_turn>).
        # These tokens tend to act as "summary" positions in the residual stream.
        assert input_ids is not None and special_ids is not None
        idx = answer_start - 1
        while idx >= 0 and int(input_ids[idx]) not in special_ids:
            idx -= 1
        if idx < 0:
            raise ValueError("No special token found in prompt before answer_start")
        mask[idx] = True
    else:
        raise AssertionError(f"Unhandled mask strategy: {strategy!r}")
    return mask


def _find_response_end(
    input_ids: torch.Tensor, answer_start: int, special_ids: set[int]
) -> int:
    """Return the end-exclusive index of the assistant response."""
    seq_len = input_ids.shape[0]
    start = max(answer_start, 0)

    for idx in range(start, seq_len):
        if int(input_ids[idx]) in special_ids:
            return idx
    return seq_len


def prepare_inputs(
    tokenizer,
    system_prompt: str,
    qa_pairs: list[QAPair],
    mask_strategy: MaskStrategy = MaskStrategy.RESPONSE_MEAN,
) -> list[PreparedInput]:
    """Format QA pairs into prompts and token masks for a given strategy.

    The formatted string from ``apply_chat_template`` already contains the
    model's BOS token, so we re-tokenize with ``add_special_tokens=False`` to
    avoid a phantom extra BOS that would shift ``answer_start`` by +1 and
    misalign the mask with the actual response tokens.

    Response masks stop before the first trailing special token after the
    assistant answer so template delimiters are not averaged.
    """
    prepared: list[PreparedInput] = []
    special_ids = set(tokenizer.all_special_ids)
    for qa in qa_pairs:
        messages = _build_messages(qa, system_prompt)
        full_prompt, answer_start = format_messages(messages, tokenizer)
        input_ids = tokenizer(
            full_prompt, return_tensors="pt", add_special_tokens=False
        ).input_ids[0]
        answer_end = _find_response_end(input_ids, answer_start, special_ids)
        mask = _build_mask(
            input_ids.shape[0],
            answer_start,
            answer_end,
            mask_strategy,
            input_ids=input_ids,
            special_ids=special_ids,
        )
        prepared.append(
            PreparedInput(
                question=qa.question,
                input_ids=input_ids,
                answer_start=answer_start,
                token_mask=mask,
            )
        )
    return prepared


def _render_sample(p: PreparedInput, tokenizer, max_tokens: int = 200) -> Text:
    ids = p.input_ids.tolist()
    tokens = tokenizer.convert_ids_to_tokens(ids)
    special_ids = set(tokenizer.all_special_ids)

    if len(ids) > max_tokens:
        head = max_tokens // 2
        tail = max_tokens - head
        indices = list(range(head)) + [None] + list(range(len(ids) - tail, len(ids)))
    else:
        indices = list(range(len(ids)))

    rendered = Text()
    for idx in indices:
        if idx is None:
            rendered.append(" … ", style="dim")
            continue
        raw = tokens[idx].replace("▁", " ").replace("Ċ", "\n")
        if p.token_mask[idx]:
            style = _MASK_STYLE
        elif ids[idx] in special_ids:
            style = _SPECIAL_STYLE
        elif idx >= p.answer_start:
            style = _RESPONSE_STYLE
        else:
            style = _PROMPT_STYLE
        parts = raw.split("\n")
        for part_idx, part in enumerate(parts):
            if part:
                rendered.append(part, style=style)
            if part_idx < len(parts) - 1:
                rendered.append("\n")
    return rendered


def preview_prepared_inputs(
    prepared: list[PreparedInput],
    tokenizer,
    *,
    variant: str | None = None,
    mask_strategy: MaskStrategy | None = None,
    max_tokens: int = 200,
) -> None:
    """Pretty-print prepared inputs with masked tokens highlighted."""

    console = Console()
    title = ["extraction preview"]
    if variant is not None:
        title.append(f"variant=[bold]{variant}[/]")
    if mask_strategy is not None:
        title.append(f"strategy=[magenta]{mask_strategy}[/]")
    console.rule(" — ".join(title))

    for i, p in enumerate(prepared):
        question = p.question if len(p.question) <= 80 else p.question[:77] + "..."
        meta = Text.from_markup(
            f"[dim]question=[/]{question!r}\n"
            f"[dim]seq_len=[/]{int(p.input_ids.shape[0])}  "
            f"[dim]answer_start=[/]{p.answer_start}  "
            f"[dim]masked_tokens=[/]{int(p.token_mask.sum())}"
        )
        body = Text()
        body.append_text(meta)
        body.append("\n\n")
        body.append_text(_render_sample(p, tokenizer, max_tokens))
        console.print(Panel(body, title=f"sample {i}", border_style="blue"))


def _free_memory() -> None:
    """Release temporary tensors between variants to keep memory bounded."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def run_extraction(
    model: StandardizedTransformer,
    model_name: str,
    persona: PersonaData,
    qa_pairs: list[QAPair],
    variants: tuple[str, ...],
    mask_strategy: MaskStrategy = MaskStrategy.RESPONSE_MEAN,
    remote: bool = False,
    on_status: Callable | None = None,
    verbose: bool = False,
) -> list[ExtractionResult]:
    """Extract and save per-question activation vectors for each prompt variant.

    Args:
        model: Loaded standardized nnterp model.
        model_name: HuggingFace model identifier used for artifact paths.
        persona: The persona whose QA pairs are being extracted.
        qa_pairs: Question-answer pairs to run extraction on.
        variants: Prompt variants to extract (``"templated"`` or ``"biography"``).
        mask_strategy: Which tokens should contribute to the averaged hidden
            state. See :class:`MaskStrategy`.
        remote: Whether to execute on NDIF.
        on_status: Forwarded to extract_activations. Called on each NDIF status
            update with (job_id, status_name, description).
        verbose: If True, print a rich preview of each prepared sample before
            the forward pass.

    Returns:
        One ExtractionResult per variant.
    """
    if not qa_pairs:
        raise ValueError("No QA pairs selected for extraction")
    if invalid := set(variants) - set(SUPPORTED_VARIANTS):
        raise ValueError(f"Unsupported variants: {invalid}")

    store = ActivationStore(model_name)
    results: list[ExtractionResult] = []

    for variant in variants:
        if variant == "baseline":
            system_prompt = format_roleplay_prompt(mode="mc")
        else:
            system_prompt = format_roleplay_prompt(
                getattr(persona, f"{variant}_view"), mode="mc"
            )
        prepared = prepare_inputs(
            tokenizer=model.tokenizer,
            system_prompt=system_prompt,
            qa_pairs=qa_pairs,
            mask_strategy=mask_strategy,
        )

        if verbose:
            preview_prepared_inputs(
                prepared,
                tokenizer=model.tokenizer,
                variant=variant,
                mask_strategy=mask_strategy,
            )

        vectors = extract_activations(
            model,
            input_ids_list=[p.input_ids for p in prepared],
            token_masks=[p.token_mask for p in prepared],
            remote=remote,
            on_status=on_status,
        )

        artifact_dir = store.save(
            variant,
            persona.id,
            persona.name,
            vectors,
            [p.question for p in prepared],
        )

        results.append(
            ExtractionResult(
                variant=variant,
                output_dir=artifact_dir,
                n_questions=vectors.shape[0],
                persona_name=persona.name,
            )
        )

        del vectors, prepared
        _free_memory()

    return results
