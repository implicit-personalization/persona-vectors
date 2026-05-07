import gc
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Callable, Literal, cast

import torch
from nnterp import StandardizedTransformer
from persona_data.prompts import (
    format_mc_question,
    format_messages,
    format_prompt,
    mc_correct_letter,
)
from persona_data.synth_persona import (
    BASELINE_PERSONA_ID,
    PersonaData,
    QAPair,
    SynthPersonaDataset,
)
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from persona_vectors.activations import extract_activations
from persona_vectors.artifacts import SUPPORTED_VARIANTS, ActivationStore

_MASK_BG = "on green"
_TEMPLATE_STYLE = "dim"
_QUESTION_STYLE = "yellow"
_RESPONSE_STYLE = "bright_cyan"
_SPECIAL_STYLE = "bold magenta"


class MaskStrategy(StrEnum):
    """Which tokens contribute to the averaged hidden state.

    ``persona_*`` averages the persona/system-prompt prefix, ``question_*``
    targets the user question, and ``answer_*`` targets the assistant answer
    or the token immediately before it.
    """

    PERSONA_MEAN = "persona_mean"
    PERSONA_LAST = "persona_last"
    QUESTION_LAST = "question_last"
    QUESTION_LAST_SPECIAL = "question_last_special"
    ANSWER_PREVIOUS = "answer_previous"
    ANSWER_FIRST = "answer_first"
    ANSWER_LAST = "answer_last"
    ANSWER_MEAN = "answer_mean"


@dataclass(frozen=True)
class Span:
    char_start: int
    char_end: int
    token_start: int
    token_end: int


@dataclass(frozen=True)
class PromptSpans:
    template: Span
    question: Span
    response: Span


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
        sample_id: Stable dataset sample id, used to verify alignment across
            prompt variants.
        question: Original user question text.
        prompt_text: Fully rendered chat prompt used for tokenization.
        spans: Semantic token/character spans for template, question, answer.
        offset_mapping: Token-level character offsets for ``prompt_text``.
        input_ids: Token ids for the formatted prompt (shape ``(seq_len,)``).
        token_mask: Boolean mask over ``input_ids`` selecting which tokens
            contribute to the averaged hidden state.
    """

    sample_id: str
    question: str
    prompt_text: str
    spans: PromptSpans
    offset_mapping: list[tuple[int, int]]
    input_ids: torch.Tensor
    token_mask: torch.Tensor


@dataclass(frozen=True)
class TokenSegment:
    text: str
    role: str
    is_special: bool
    is_masked: bool


def _build_mask(
    seq_len: int,
    spans: PromptSpans,
    strategy: MaskStrategy,
    input_ids: torch.Tensor,
    special_ids: set[int],
) -> torch.Tensor:
    """Return a boolean mask over ``seq_len`` tokens for the given strategy."""
    answer_start = spans.response.token_start
    answer_end = spans.response.token_end
    if answer_start >= seq_len:
        raise ValueError(f"Invalid answer_start={answer_start} for seq_len={seq_len}")
    if answer_end <= answer_start or answer_end > seq_len:
        raise ValueError(
            f"Invalid answer_end={answer_end} for answer_start={answer_start} and seq_len={seq_len}"
        )

    mask = torch.zeros(seq_len, dtype=torch.bool)
    if strategy is MaskStrategy.PERSONA_MEAN:
        mask[spans.template.token_start : spans.template.token_end] = True
    elif strategy is MaskStrategy.PERSONA_LAST:
        mask[spans.template.token_end - 1] = True
    elif strategy is MaskStrategy.ANSWER_MEAN:
        mask[answer_start:answer_end] = True
    elif strategy is MaskStrategy.ANSWER_PREVIOUS:
        idx = answer_start - 1
        if idx < 0:
            raise ValueError("Expected a token immediately before answer span")
        mask[idx] = True
    elif strategy is MaskStrategy.ANSWER_FIRST:
        mask[answer_start] = True
    elif strategy is MaskStrategy.ANSWER_LAST:
        mask[answer_end - 1] = True
    elif strategy is MaskStrategy.QUESTION_LAST:
        mask[spans.question.token_end - 1] = True
    elif strategy is MaskStrategy.QUESTION_LAST_SPECIAL:
        idx = spans.question.token_end
        if idx >= seq_len or int(input_ids[idx]) not in special_ids:
            raise ValueError("Expected a special token immediately after question span")
        mask[idx] = True
    else:
        raise AssertionError(f"Unhandled mask strategy: {strategy!r}")
    return mask


def _find_text_span(text: str, needle: str, start: int = 0) -> tuple[int, int]:
    """Find ``needle`` in ``text`` at or after ``start``.

    We try the exact text first, then a stripped version to account for chat
    templates that trim message content before rendering.
    """
    for candidate in (needle, needle.strip()):
        if not candidate:
            continue
        idx = text.find(candidate, start)
        if idx >= 0:
            return idx, idx + len(candidate)
    raise ValueError(f"Could not find span for {needle!r} in rendered prompt")


def _char_span_to_token_span(
    offsets: list[tuple[int, int]], char_start: int, char_end: int
) -> Span:
    token_start = None
    token_end = None
    for idx, (start, end) in enumerate(offsets):
        if token_start is None and end > char_start:
            token_start = idx
        if start < char_end:
            token_end = idx + 1

    if token_start is None or token_end is None or token_start >= token_end:
        raise ValueError(
            f"Could not map char span [{char_start}, {char_end}) onto token offsets"
        )
    return Span(
        char_start=char_start,
        char_end=char_end,
        token_start=token_start,
        token_end=token_end,
    )


def _build_prompt_spans(
    full_prompt: str,
    offsets: list[tuple[int, int]],
    template_text: str,
    question_text: str,
    response_text: str,
) -> PromptSpans:
    template_char = _find_text_span(full_prompt, template_text, 0)
    question_char = _find_text_span(full_prompt, question_text, template_char[1])
    response_char = _find_text_span(full_prompt, response_text, question_char[1])

    return PromptSpans(
        template=_char_span_to_token_span(offsets, *template_char),
        question=_char_span_to_token_span(offsets, *question_char),
        response=_char_span_to_token_span(offsets, *response_char),
    )


def _token_role(p: PreparedInput, token_idx: int) -> str:
    if p.spans.response.token_start <= token_idx < p.spans.response.token_end:
        return "response"
    if p.spans.question.token_start <= token_idx < p.spans.question.token_end:
        return "question"
    if p.spans.template.token_start <= token_idx < p.spans.template.token_end:
        return "template"
    return "other"


def preview_token_segments(
    p: PreparedInput, tokenizer, *, max_tokens: int = 200
) -> list[TokenSegment]:
    """Return token preview segments without tying rendering to Rich or HTML."""
    seq_len = int(p.input_ids.shape[0])
    special_ids = set(tokenizer.all_special_ids)
    head = max_tokens if max_tokens > 0 else seq_len
    tail = 8 if max_tokens <= 0 else max(8, max_tokens // 4)
    answer_extra = 8 if max_tokens <= 0 else max(8, max_tokens // 4)
    prefix_end = min(p.spans.template.token_start + head, seq_len)
    tail_start = min(max(prefix_end, p.spans.template.token_end - tail), seq_len)
    answer_end = min(seq_len, p.spans.response.token_end + answer_extra)

    indices: list[int | None] = list(range(0, prefix_end))
    if prefix_end < tail_start:
        indices.append(None)
    indices.extend(range(tail_start, answer_end))
    if answer_end < seq_len:
        indices.append(None)

    segments: list[TokenSegment] = []
    for token_idx in indices:
        if token_idx is None:
            segments.append(TokenSegment(" … ", "other", False, False))
            continue
        token_char_start, token_char_end = p.offset_mapping[token_idx]
        text = p.prompt_text[token_char_start:token_char_end]
        if not text:
            text = tokenizer.convert_ids_to_tokens([int(p.input_ids[token_idx])])[0]
        is_special = int(p.input_ids[token_idx]) in special_ids
        segments.append(
            TokenSegment(
                text=text,
                role=_token_role(p, token_idx),
                is_special=is_special,
                is_masked=bool(p.token_mask[token_idx]),
            )
        )
    return segments


def _segment_style(segment: TokenSegment) -> str:
    styles = {
        "response": _RESPONSE_STYLE,
        "question": _QUESTION_STYLE,
        "template": _TEMPLATE_STYLE,
    }
    style = styles.get(segment.role, "dim")
    if segment.is_special:
        style = _SPECIAL_STYLE
    if segment.is_masked:
        style = f"{style} {_MASK_BG}"
    return style


def _render_prompt_preview(p: PreparedInput, tokenizer, max_tokens: int = 200) -> Text:
    rendered = Text()
    for segment in preview_token_segments(p, tokenizer, max_tokens=max_tokens):
        rendered.append(segment.text, style=_segment_style(segment))
    return rendered


def prepare_inputs(
    tokenizer,
    system_prompt: str,
    qa_pairs: list[QAPair],
    mask_strategy: MaskStrategy = MaskStrategy.ANSWER_MEAN,
) -> list[PreparedInput]:
    """Format QA pairs into prompts and token masks for a given strategy.

    The formatted string from ``apply_chat_template`` already contains the
    model's BOS token, so we re-tokenize with ``add_special_tokens=False`` to
    avoid a phantom extra BOS that would shift ``answer_start`` by +1 and
    misalign the mask with the actual response tokens.

    Persona masks only cover the system-prompt prefix, not the question.
    Response masks stop before the first trailing special token after the
    assistant answer so template delimiters are not averaged.
    """
    prepared: list[PreparedInput] = []
    special_ids = set(tokenizer.all_special_ids)
    for qa in qa_pairs:
        if qa.item_type == "mcq" and qa.correct_choice_index is not None:
            user_content = format_mc_question(qa)
            answer_content = mc_correct_letter(qa)
        else:
            user_content = qa.question
            answer_content = qa.answer

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": answer_content},
        ]
        full_prompt, _ = format_messages(messages, tokenizer)
        encoding = tokenizer(
            full_prompt,
            return_offsets_mapping=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = encoding.input_ids[0]
        offsets = [tuple(pair) for pair in encoding.offset_mapping[0].tolist()]
        spans = _build_prompt_spans(
            full_prompt,
            offsets,
            system_prompt,
            messages[1]["content"],
            messages[2]["content"],
        )
        mask = _build_mask(
            input_ids.shape[0],
            spans,
            mask_strategy,
            input_ids=input_ids,
            special_ids=special_ids,
        )
        prepared.append(
            PreparedInput(
                sample_id=qa.qid,
                question=qa.question,
                prompt_text=full_prompt,
                spans=spans,
                offset_mapping=offsets,
                input_ids=input_ids,
                token_mask=mask,
            )
        )
    return prepared


def prepare_inputs_for_strategy(
    tokenizer,
    system_prompt: str,
    qa_pairs: list[QAPair],
    mask_strategy: MaskStrategy,
) -> list[PreparedInput]:
    """Prepare the smallest useful batch for the requested mask strategy.

    Persona masks (``persona_mean`` / ``persona_last``) ignore QA tokens, so
    we only run the first QA pair through the full pipeline — the resulting
    template-span vector is identical no matter which question is paired.
    """
    if not qa_pairs:
        raise ValueError("No QA pairs selected for extraction")

    persona_only = mask_strategy in (
        MaskStrategy.PERSONA_MEAN,
        MaskStrategy.PERSONA_LAST,
    )
    selected = qa_pairs[:1] if persona_only else qa_pairs
    return prepare_inputs(
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        qa_pairs=selected,
        mask_strategy=mask_strategy,
    )


def _render_sample(p: PreparedInput, tokenizer, max_tokens: int = 200) -> Text:
    rendered = Text()
    question = p.question if len(p.question) <= 80 else p.question[:77] + "..."
    rendered.append("question=", style="dim")
    rendered.append(repr(question))
    rendered.append("\n")
    rendered.append("seq_len=", style="dim")
    rendered.append(str(int(p.input_ids.shape[0])))
    rendered.append("  masked_tokens=", style="dim")
    rendered.append(str(int(p.token_mask.sum())))

    rendered.append("\n\n")
    rendered.append_text(_render_prompt_preview(p, tokenizer, max_tokens))
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
        console.print(
            Panel(
                _render_sample(p, tokenizer, max_tokens),
                title=f"sample {i}",
                border_style="blue",
            )
        )


def select_personas_with_qa(
    dataset: SynthPersonaDataset,
    persona_id: str | None = None,
    *,
    include_baseline: bool = True,
) -> list[tuple[PersonaData, list[QAPair]]]:
    """Select (persona, qa_pairs) tuples for extraction.

    This function only selects from personas already loaded in ``dataset``. If
    ``SynthPersonaDataset(sample_size=...)`` sliced out the Assistant baseline,
    ``include_baseline=True`` cannot recover it; load the dataset without that
    slice, or pass ``persona_id=BASELINE_PERSONA_ID`` to a full dataset.

    - ``persona_id=None``: all loaded personas, optionally including the loaded
      Assistant baseline.
    - ``persona_id=<id>``: just that loaded persona. ``include_baseline`` is
      ignored.

    QA pairs come from ``train_test_split`` for normal personas and from the
    shared MCQ pool for the baseline (which has no per-persona train split).
    Normal personas use a 50-question train cap here to keep remote extraction
    bounded. Personas with no matching QA pairs are skipped.
    """
    if persona_id is not None:
        if persona_id == BASELINE_PERSONA_ID and dataset.baseline is not None:
            personas = [dataset.baseline]
        else:
            match = next((p for p in dataset if p.id == persona_id), None)
            if match is None:
                raise ValueError(f"No persona found with id {persona_id!r}")
            personas = [match]
    else:
        personas = list(dataset)
        if (
            include_baseline
            and dataset.baseline is not None
            and not any(p.id == BASELINE_PERSONA_ID for p in personas)
        ):
            personas.append(dataset.baseline)

    runs: list[tuple[PersonaData, list[QAPair]]] = []
    for persona in personas:
        if persona.id == BASELINE_PERSONA_ID:
            qa_pairs = list(
                dataset.get_qa(BASELINE_PERSONA_ID, item_type="mcq", scope="shared")
            )
        else:
            qa_pairs, _ = dataset.train_test_split(persona.id, n_train=50)
        if qa_pairs:
            runs.append((persona, qa_pairs))
    return runs


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
    qa_pairs: list[QAPair],
    variants: tuple[str, ...],
    *,
    persona: PersonaData | None = None,
    mask_strategy: MaskStrategy = MaskStrategy.ANSWER_MEAN,
    remote: bool = False,
    on_status: Callable | None = None,
    verbose: bool = False,
    activations_dir: str | Path | None = None,
) -> list[ExtractionResult]:
    """Extract and save a mean activation vector per variant for the persona.

    Runs a forward pass for each QA pair, computes the masked-token mean
    hidden state at each layer, then averages across all QA pairs before
    saving. The saved artifact is a single ``(num_layers, hidden_size)``
    tensor per persona per variant.

    Each variant reads ``<variant>_view`` from ``persona``. The Assistant
    baseline is just another ``PersonaData`` with the usual views.

    Args:
        model: Loaded standardized nnterp model.
        model_name: HuggingFace model identifier used for artifact paths.
        qa_pairs: Question-answer pairs to run extraction on.
        variants: Variants to extract. See ``SUPPORTED_VARIANTS``.
        persona: Persona used to render the requested variants.
        mask_strategy: Which tokens should contribute to the averaged hidden
            state. See :class:`MaskStrategy`.
        remote: Whether to execute on NDIF.
        on_status: Forwarded to extract_activations. Called on each NDIF status
            update with (job_id, status_name, description).
        verbose: If True, print a rich preview of each prepared sample before
            the forward pass.
        activations_dir: Root directory for saved activations. Pass a unique
            subdirectory to keep multiple runs separate.

    Returns:
        One ExtractionResult per variant, in the order requested.
    """
    if invalid := set(variants) - set(SUPPORTED_VARIANTS):
        raise ValueError(f"Unsupported variants: {invalid}")

    store = ActivationStore(model_name, root_dir=activations_dir)
    results: list[ExtractionResult] = []

    if persona is None:
        raise ValueError("run_extraction requires a PersonaData")

    for variant in variants:
        persona_variant = cast(Literal["templated", "biography"], variant)
        system_prompt = format_prompt(persona, persona_variant)
        persona_id, persona_name = persona.id, persona.name

        prepared = prepare_inputs_for_strategy(
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
            persona_id,
            persona_name,
            vectors,
            [p.sample_id for p in prepared],
            mask_strategy=mask_strategy,
        )

        results.append(
            ExtractionResult(
                variant=variant,
                output_dir=artifact_dir,
                n_questions=len(prepared),
                persona_name=persona_name,
            )
        )

        del vectors, prepared
        _free_memory()

    return results
