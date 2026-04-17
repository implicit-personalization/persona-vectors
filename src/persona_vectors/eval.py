from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from nnterp import StandardizedTransformer
from persona_data.prompts import format_mc_question, format_roleplay_prompt, normalize_messages
from persona_data.synth_persona import PersonaData, QAPair

ConditionName = str


@dataclass
class ChoiceEvalResult:
    persona_id: str
    persona_name: str
    qid: str
    question: str
    qa_type: str
    condition: ConditionName
    gold_letter: str
    predicted_letter: str
    correct: bool
    gold_prob: float
    gold_logprob: float
    margin_vs_best_other: float
    choice_letters: list[str]
    choice_probs: list[float]
    choice_logprobs: list[float]
    steering_layer: int | None
    steering_alpha: float | None

    def to_dict(self) -> dict:
        return asdict(self)


def build_system_prompt(persona: PersonaData, condition: ConditionName) -> str:
    if condition == "bare" or condition == "steered":
        return format_roleplay_prompt(mode="mc")
    if condition == "templated":
        return format_roleplay_prompt(persona.templated_view, mode="mc")
    if condition == "biography":
        return format_roleplay_prompt(persona.biography_view, mode="mc")
    raise ValueError(f"Unsupported condition: {condition}")


def build_generation_prompt(tokenizer, system_prompt: str, qa: QAPair) -> tuple[str, int]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": format_mc_question(qa)},
    ]
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        prompt = tokenizer.apply_chat_template(
            normalize_messages(messages),
            tokenize=False,
            add_generation_prompt=True,
        )

    prompt_ids = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids[0]
    return prompt, int(prompt_ids.shape[0])


def choice_token_ids(tokenizer, qa: QAPair) -> tuple[list[str], list[int]]:
    letters = [chr(ord("A") + i) for i in range(len(qa.choices))]
    token_ids: list[int] = []
    for letter in letters:
        ids = tokenizer(letter, add_special_tokens=False).input_ids
        if len(ids) != 1:
            raise ValueError(
                f"Expected single-token MC label for {letter!r}, got ids={ids!r}"
            )
        token_ids.append(int(ids[0]))
    return letters, token_ids


def score_choice_distribution(
    model: StandardizedTransformer,
    prompt: str,
    prompt_len: int,
    choice_token_ids: list[int],
    *,
    remote: bool,
    steering_layer: int | None = None,
    steering_vector: torch.Tensor | None = None,
    steering_alpha: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = model.tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids[0]

    with torch.no_grad(), model.trace(input_ids.unsqueeze(0), remote=remote):
        if steering_vector is not None:
            if steering_layer is None or steering_alpha is None:
                raise ValueError("steering_layer and steering_alpha are required")
            model.steer(
                layers=steering_layer,
                steering_vector=steering_vector,
                factor=steering_alpha,
                token_positions=prompt_len - 1,
            )

        choice_logits = model.logits[0, prompt_len - 1, choice_token_ids].float()
        choice_logprobs = torch.log_softmax(choice_logits, dim=-1).save()

    if hasattr(choice_logprobs, "value") and getattr(choice_logprobs, "value") is not None:
        choice_logprobs = choice_logprobs.value
    if not isinstance(choice_logprobs, torch.Tensor):
        raise TypeError(
            f"choice scoring did not resolve to a tensor: {type(choice_logprobs)!r}"
        )
    choice_logprobs = choice_logprobs.detach().cpu()
    choice_probs = choice_logprobs.exp()
    return choice_logprobs, choice_probs


def evaluate_mc_question(
    model: StandardizedTransformer,
    persona: PersonaData,
    qa: QAPair,
    condition: ConditionName,
    *,
    remote: bool,
    steering_layer: int | None = None,
    steering_vector: torch.Tensor | None = None,
    steering_alpha: float | None = None,
) -> ChoiceEvalResult:
    if qa.answer_format != "choice" or qa.correct_choice_index is None:
        raise ValueError(f"QAPair {qa.qid!r} is not a scored multiple-choice item")

    system_prompt = build_system_prompt(persona, condition)
    prompt, prompt_len = build_generation_prompt(model.tokenizer, system_prompt, qa)
    choice_letters, token_ids = choice_token_ids(model.tokenizer, qa)
    choice_logprobs, choice_probs = score_choice_distribution(
        model,
        prompt,
        prompt_len,
        token_ids,
        remote=remote,
        steering_layer=steering_layer,
        steering_vector=steering_vector,
        steering_alpha=steering_alpha,
    )

    gold_idx = qa.correct_choice_index
    top_idx = int(choice_logprobs.argmax().item())
    other_idxs = [i for i in range(len(choice_letters)) if i != gold_idx]
    best_other = (
        choice_logprobs[other_idxs].max().item() if other_idxs else float("-inf")
    )

    return ChoiceEvalResult(
        persona_id=persona.id,
        persona_name=persona.name,
        qid=qa.qid,
        question=qa.question,
        qa_type=qa.type,
        condition=condition,
        gold_letter=choice_letters[gold_idx],
        predicted_letter=choice_letters[top_idx],
        correct=top_idx == gold_idx,
        gold_prob=float(choice_probs[gold_idx].item()),
        gold_logprob=float(choice_logprobs[gold_idx].item()),
        margin_vs_best_other=float(choice_logprobs[gold_idx].item() - best_other),
        choice_letters=choice_letters,
        choice_probs=[float(v) for v in choice_probs.tolist()],
        choice_logprobs=[float(v) for v in choice_logprobs.tolist()],
        steering_layer=steering_layer,
        steering_alpha=steering_alpha,
    )
