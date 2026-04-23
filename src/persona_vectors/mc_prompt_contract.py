from __future__ import annotations

from typing import Literal

from persona_data.prompts import format_messages, format_mc_question
from persona_data.synth_persona import PersonaData, QAPair

MC_RENDER_STYLE = Literal["system_user", "inline_user"]
MCCondition = Literal["bare", "steered", "baseline", "templated", "biography"]

MC_INSTRUCTION_BLOCK = (
    "You are answering a multiple-choice question.\n"
    "If the context describes a specific person, answer by roleplaying as that person.\n"
    "If no person context is provided, answer using only the question and options.\n"
    "Choose the one option that best matches what you would most likely think, prefer, say, or do.\n"
    "Use only the provided context.\n"
    "If the context is not sufficient to support one substantive option, choose E.\n"
    "Return exactly one uppercase letter: A, B, C, D, or E."
)

MC_PROMPT_CONTRACT_VERSION = "synth_persona_mc_v1_20260422"

MC_CONTEXT_HEADERS = {
    "bare": "NO PERSON CONTEXT",
    "biography": "PERSON BIOGRAPHY",
    "attributes": "PERSON ATTRIBUTES",
}

DEFAULT_UNSURE_OPTION = "Not enough information from the context."


def tokenizer_supports_system_role(tokenizer) -> bool:
    if getattr(tokenizer, "chat_template", None) is None:
        raise ValueError("tokenizer.chat_template is missing; cannot infer MC render style")
    try:
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "a"},
                {"role": "user", "content": "b"},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return False
    return True


def infer_mc_render_style(tokenizer) -> MC_RENDER_STYLE:
    return "system_user" if tokenizer_supports_system_role(tokenizer) else "inline_user"


def _context_for_condition(persona: PersonaData, condition: MCCondition) -> tuple[str, str]:
    if condition in ("bare", "steered", "baseline"):
        return MC_CONTEXT_HEADERS["bare"], ""
    if condition == "templated":
        return MC_CONTEXT_HEADERS["attributes"], persona.templated_view.strip()
    if condition == "biography":
        return MC_CONTEXT_HEADERS["biography"], persona.biography_view.strip()
    raise ValueError(f"Unsupported MC condition: {condition}")


def _substantive_options_and_unsure(qa: QAPair) -> tuple[list[str], str]:
    cleaned = [str(option).strip() for option in qa.choices if str(option).strip()]
    if len(cleaned) < 4:
        raise ValueError(f"Expected at least 4 substantive options for {qa.qid!r}")
    substantive = cleaned[:4]
    unsure = cleaned[4] if len(cleaned) >= 5 else DEFAULT_UNSURE_OPTION
    return substantive, unsure


def _render_mc_user_prompt(
    *,
    context_header: str,
    context_body: str,
    question: str,
    options: list[str],
    unsure_option: str,
) -> str:
    option_lines = [
        f"{label}. {text}"
        for label, text in zip(["A", "B", "C", "D"], options, strict=True)
    ]
    option_lines.append(f"E. {unsure_option}")
    return (
        f"{context_header}:\n"
        f"{context_body}\n\n"
        "QUESTION:\n"
        f"{question.strip()}\n\n"
        "OPTIONS:\n"
        f"{chr(10).join(option_lines)}\n\n"
        "Answer with one letter only."
    )


def render_mc_messages(
    tokenizer,
    *,
    persona: PersonaData,
    qa: QAPair,
    condition: MCCondition,
    answer: str | None = None,
) -> list[dict[str, str]]:
    if qa.answer_format != "choice" or qa.correct_choice_index is None:
        raise ValueError(f"MC prompt contract only supports scored choice QA pairs, got {qa.qid!r}")
    context_header, context_body = _context_for_condition(persona, condition)
    options, unsure_option = _substantive_options_and_unsure(qa)
    user_prompt = _render_mc_user_prompt(
        context_header=context_header,
        context_body=context_body,
        question=qa.question,
        options=options,
        unsure_option=unsure_option,
    )
    render_style = infer_mc_render_style(tokenizer)
    if render_style == "system_user":
        messages = [
            {"role": "system", "content": MC_INSTRUCTION_BLOCK},
            {"role": "user", "content": user_prompt},
        ]
    else:
        messages = [
            {"role": "user", "content": f"{MC_INSTRUCTION_BLOCK}\n\n{user_prompt}"},
        ]
    if answer is not None:
        messages.append({"role": "assistant", "content": answer})
    return messages


def render_mc_generation_prompt(
    tokenizer,
    *,
    persona: PersonaData,
    qa: QAPair,
    condition: MCCondition,
) -> tuple[str, int]:
    messages = render_mc_messages(tokenizer, persona=persona, qa=qa, condition=condition)
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids[0]
    return prompt, int(prompt_ids.shape[0])


def render_mc_prompt_with_answer(
    tokenizer,
    *,
    persona: PersonaData,
    qa: QAPair,
    condition: MCCondition,
    answer: str,
) -> tuple[str, int]:
    messages = render_mc_messages(
        tokenizer,
        persona=persona,
        qa=qa,
        condition=condition,
        answer=answer,
    )
    return format_messages(messages, tokenizer)
