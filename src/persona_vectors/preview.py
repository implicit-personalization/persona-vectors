from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from persona_vectors.extraction import MaskStrategy, PreparedInput

_MASK_BG = "on green"
_TEMPLATE_STYLE = "dim"
_QUESTION_STYLE = "yellow"
_RESPONSE_STYLE = "bright_cyan"
_SPECIAL_STYLE = "bold magenta"


@dataclass(frozen=True)
class TokenSegment:
    text: str
    role: str
    is_special: bool
    is_masked: bool


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
            segments.append(TokenSegment(" ... ", "other", False, False))
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


def _render_prompt_preview(p: PreparedInput, tokenizer, max_tokens: int = 200):
    from rich.text import Text

    rendered = Text()
    for segment in preview_token_segments(p, tokenizer, max_tokens=max_tokens):
        rendered.append(segment.text, style=_segment_style(segment))
    return rendered


def _render_sample(p: PreparedInput, tokenizer, max_tokens: int = 200):
    from rich.text import Text

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
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    title = ["extraction preview"]
    if variant is not None:
        title.append(f"variant=[bold]{variant}[/]")
    if mask_strategy is not None:
        title.append(f"strategy=[magenta]{mask_strategy}[/]")
    console.rule(" - ".join(title))

    for i, p in enumerate(prepared):
        console.print(
            Panel(
                _render_sample(p, tokenizer, max_tokens),
                title=f"sample {i}",
                border_style="blue",
            )
        )
