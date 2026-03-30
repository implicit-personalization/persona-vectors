from dataclasses import dataclass
from typing import Literal

import torch

from src.prompt_format import (
    format_biography_prompt,
    format_templated_prompt,
    normalize_messages,
)
from src.synth_persona_io import PersonaData

SystemPromptMode = Literal["empty", "templated", "biography"]


@dataclass
class ChatReply:
    text: str
    prompt_tokens: int
    output_tokens: int
    past_key_values: object | None


def resolve_system_prompt(
    persona: PersonaData | None,
    mode: SystemPromptMode,
) -> str:
    """Resolve the active system prompt for chat.

    Args:
        persona: Selected persona, if any.
        mode: Prompt mode selected in the UI.

    Returns:
        The rendered system prompt string.
    """

    if persona is None:
        return ""

    if mode == "templated":
        return format_templated_prompt(persona.templated_prompt)
    if mode == "biography":
        return format_biography_prompt(persona.biography_md)
    return ""


def _format_plain_messages(
    messages: list[dict[str, str]], add_generation_prompt: bool
) -> str:
    lines: list[str] = []

    for message in messages:
        role = message["role"]
        content = message["content"]

        if role == "system":
            if content:
                lines.append(f"System: {content}")
        elif role == "user":
            lines.append(f"User: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")
        else:
            lines.append(f"{role.title()}: {content}")

    if add_generation_prompt and (not lines or not lines[-1].startswith("Assistant:")):
        lines.append("Assistant:")

    return "\n\n".join(lines)


def _format_generation_prompt(
    messages: list[dict[str, str]], tokenizer
) -> tuple[str, int]:
    normalized_messages = messages

    try:
        prompt = tokenizer.apply_chat_template(
            normalized_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        normalized_messages = normalize_messages(messages)

        try:
            prompt = tokenizer.apply_chat_template(
                normalized_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            prompt = _format_plain_messages(
                normalized_messages,
                add_generation_prompt=True,
            )

    prompt_token_count = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
    return prompt, prompt_token_count


def generate_chat_reply(
    model,
    messages: list[dict[str, str]],
    remote: bool,
    past_key_values: object | None = None,
    max_new_tokens: int = 256,
) -> ChatReply:
    """Generate one assistant reply from a full chat history.

    The helper uses ``model.generate`` so it works with both local and NDIF-backed
    nnsight models. The full conversation is re-rendered each turn and the cache from
    the previous turn is reused when available.

    Args:
        model: Loaded nnsight language model.
        messages: Full chat history, including any system prompt as the first message.
        remote: Whether to execute the generation on NDIF.
        past_key_values: Cache returned by the previous generation step.
        max_new_tokens: Maximum number of assistant tokens to generate.

    Returns:
        ChatReply with generated text and the updated cache.
    """

    tokenizer = model.tokenizer
    prompt, prompt_token_count = _format_generation_prompt(messages, tokenizer)

    generation_kwargs: dict[str, object] = {
        "max_new_tokens": max_new_tokens,
        "return_dict_in_generate": True,
        "use_cache": True,
    }
    if past_key_values is not None and not remote:
        generation_kwargs["past_key_values"] = past_key_values
    if remote:
        generation_kwargs["remote"] = True
        # WARNING: NDIF returns caches on CPU, so cross-turn cache reuse is not stable.

    with model.generate(prompt, **generation_kwargs) as tracer:
        generated = tracer.result.save()

    if hasattr(generated, "value") and getattr(generated, "value") is not None:
        generated = generated.value

    if not hasattr(generated, "sequences"):
        raise ValueError("Generation did not return token sequences")

    sequences = generated.sequences
    if not isinstance(sequences, torch.Tensor):
        raise TypeError("Generated sequences must be a tensor")

    generated_ids = sequences[0, prompt_token_count:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    output_tokens = int(sequences.shape[1] - prompt_token_count)

    text = text.strip()

    return ChatReply(
        text=text,
        prompt_tokens=prompt_token_count,
        output_tokens=max(0, output_tokens),
        past_key_values=(
            getattr(generated, "past_key_values", None) if not remote else None
        ),
    )
