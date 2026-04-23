from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from nnterp import StandardizedTransformer
from persona_data.synth_persona import PersonaData, QAPair
from persona_vectors.mc_prompt_contract import render_mc_generation_prompt

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
    steering_layer: int | list[int] | None
    steering_alpha: float | None

    def to_dict(self) -> dict:
        return asdict(self)


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


def _resolve_saved_tensor(tensor_like) -> torch.Tensor:
    if hasattr(tensor_like, "value") and getattr(tensor_like, "value") is not None:
        tensor_like = tensor_like.value
    if not isinstance(tensor_like, torch.Tensor):
        raise TypeError(f"choice scoring did not resolve to a tensor: {type(tensor_like)!r}")
    return tensor_like.detach().cpu()


def _normalize_steering_inputs(
    steering_layer: int | list[int] | None,
    steering_vector: torch.Tensor | None,
    steering_alpha: float | None,
) -> tuple[list[int] | None, torch.Tensor | None]:
    if steering_vector is None:
        return None, None
    steer_vec = steering_vector.squeeze(0)
    if steering_layer is None or steering_alpha is None:
        raise ValueError("steering_layer and steering_alpha are required")
    layers = [steering_layer] if isinstance(steering_layer, int) else list(steering_layer)
    if steer_vec.ndim == 2 and steer_vec.shape[0] != len(layers):
        raise ValueError(
            f"Steering tensor has {steer_vec.shape[0]} layer vectors but {len(layers)} target layers were requested"
        )
    if steer_vec.ndim not in (1, 2):
        raise ValueError(
            f"Unsupported steering_vector shape after squeeze: {tuple(steer_vec.shape)}"
        )
    return layers, steer_vec


def score_choice_distribution(
    model: StandardizedTransformer,
    prompt: str,
    prompt_len: int,
    choice_token_ids: list[int],
    *,
    remote: bool,
    steering_layer: int | list[int] | None = None,
    steering_vector: torch.Tensor | None = None,
    steering_alpha: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = model.tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids[0]
    layers, steer_vec = _normalize_steering_inputs(
        steering_layer,
        steering_vector,
        steering_alpha,
    )

    with torch.no_grad(), model.trace(input_ids.unsqueeze(0), remote=remote):
        if steer_vec is not None:
            if steer_vec.ndim == 1:
                for layer in layers:
                    model.steer(
                        layers=layer,
                        steering_vector=steer_vec,
                        factor=steering_alpha,
                        token_positions=prompt_len - 1,
                    )
            elif steer_vec.ndim == 2:
                if steer_vec.shape[0] != len(layers):
                    raise ValueError(
                        f"Steering tensor has {steer_vec.shape[0]} layer vectors but {len(layers)} target layers were requested"
                    )
                for layer, layer_vec in zip(layers, steer_vec, strict=True):
                    model.steer(
                        layers=layer,
                        steering_vector=layer_vec,
                        factor=steering_alpha,
                        token_positions=prompt_len - 1,
                    )
            else:
                raise ValueError(
                    f"Unsupported steering_vector shape after squeeze: {tuple(steer_vec.shape)}"
                )

        choice_logits = model.logits[0, prompt_len - 1, choice_token_ids].float()
        choice_logprobs = torch.log_softmax(choice_logits, dim=-1).save()

    choice_logprobs = _resolve_saved_tensor(choice_logprobs)
    choice_probs = choice_logprobs.exp()
    return choice_logprobs, choice_probs


def score_choice_distribution_batched(
    model: StandardizedTransformer,
    prompts: list[str],
    prompt_lens: list[int],
    choice_token_ids_list: list[list[int]],
    *,
    remote: bool,
    steering_layer: int | list[int] | None = None,
    steering_vector: torch.Tensor | None = None,
    steering_alpha: float | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    if not prompts:
        return [], []
    if len(prompts) != len(prompt_lens) or len(prompts) != len(choice_token_ids_list):
        raise ValueError("prompts, prompt_lens, and choice_token_ids_list must align")

    enc = model.tokenizer(
        prompts,
        return_tensors="pt",
        add_special_tokens=False,
        padding=True,
    )
    input_ids = enc.input_ids
    attention_mask = enc.attention_mask
    token_positions = [prompt_len - 1 for prompt_len in prompt_lens]

    layers, steer_vec = _normalize_steering_inputs(
        steering_layer,
        steering_vector,
        steering_alpha,
    )
    num_choices = len(choice_token_ids_list[0])
    if any(len(choice_ids) != num_choices for choice_ids in choice_token_ids_list):
        raise ValueError("Batched MC scoring requires a fixed number of choices per example")

    with torch.no_grad(), model.trace(input_ids, attention_mask=attention_mask, remote=remote):
        if steer_vec is not None:
            batch_index = list(range(len(prompts)))
            if steer_vec.ndim == 1:
                for layer in layers:
                    model.steer(
                        layers=layer,
                        steering_vector=steer_vec,
                        factor=steering_alpha,
                        token_positions=token_positions,
                        batch_index=batch_index,
                    )
            elif steer_vec.ndim == 2:
                for layer, layer_vec in zip(layers, steer_vec, strict=True):
                    model.steer(
                        layers=layer,
                        steering_vector=layer_vec,
                        factor=steering_alpha,
                        token_positions=token_positions,
                        batch_index=batch_index,
                    )

        batch_logprobs = []
        for batch_idx, (token_position, choice_ids) in enumerate(
            zip(token_positions, choice_token_ids_list, strict=True)
        ):
            choice_logits = model.logits[batch_idx, token_position, choice_ids].float()
            batch_logprobs.append(torch.log_softmax(choice_logits, dim=-1))
        saved_logprobs = torch.stack(batch_logprobs, dim=0).save()

    resolved = _resolve_saved_tensor(saved_logprobs)
    choice_logprobs = [resolved[idx] for idx in range(resolved.shape[0])]
    choice_probs = [item.exp() for item in choice_logprobs]
    return choice_logprobs, choice_probs


def _build_choice_eval_result(
    persona: PersonaData,
    qa: QAPair,
    condition: ConditionName,
    *,
    choice_letters: list[str],
    choice_logprobs: torch.Tensor,
    choice_probs: torch.Tensor,
    steering_layer: int | list[int] | None,
    steering_alpha: float | None,
) -> ChoiceEvalResult:
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


def evaluate_mc_question(
    model: StandardizedTransformer,
    persona: PersonaData,
    qa: QAPair,
    condition: ConditionName,
    *,
    remote: bool,
    steering_layer: int | list[int] | None = None,
    steering_vector: torch.Tensor | None = None,
    steering_alpha: float | None = None,
) -> ChoiceEvalResult:
    if qa.answer_format != "choice" or qa.correct_choice_index is None:
        raise ValueError(f"QAPair {qa.qid!r} is not a scored multiple-choice item")
    prompt, prompt_len = render_mc_generation_prompt(
        model.tokenizer,
        persona=persona,
        qa=qa,
        condition=condition,
    )
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
    return _build_choice_eval_result(
        persona,
        qa,
        condition,
        choice_letters=choice_letters,
        choice_logprobs=choice_logprobs,
        choice_probs=choice_probs,
        steering_layer=steering_layer,
        steering_alpha=steering_alpha,
    )


def evaluate_mc_questions_batch(
    model: StandardizedTransformer,
    persona: PersonaData,
    qa_pairs: list[QAPair],
    condition: ConditionName,
    *,
    remote: bool,
    steering_layer: int | list[int] | None = None,
    steering_vector: torch.Tensor | None = None,
    steering_alpha: float | None = None,
) -> list[ChoiceEvalResult]:
    if not qa_pairs:
        return []
    for qa in qa_pairs:
        if qa.answer_format != "choice" or qa.correct_choice_index is None:
            raise ValueError(f"QAPair {qa.qid!r} is not a scored multiple-choice item")

    prompts: list[str] = []
    prompt_lens: list[int] = []
    choice_letters_list: list[list[str]] = []
    choice_ids_list: list[list[int]] = []
    for qa in qa_pairs:
        prompt, prompt_len = render_mc_generation_prompt(
            model.tokenizer,
            persona=persona,
            qa=qa,
            condition=condition,
        )
        choice_letters, token_ids = choice_token_ids(model.tokenizer, qa)
        prompts.append(prompt)
        prompt_lens.append(prompt_len)
        choice_letters_list.append(choice_letters)
        choice_ids_list.append(token_ids)

    choice_logprobs_list, choice_probs_list = score_choice_distribution_batched(
        model,
        prompts,
        prompt_lens,
        choice_ids_list,
        remote=remote,
        steering_layer=steering_layer,
        steering_vector=steering_vector,
        steering_alpha=steering_alpha,
    )

    rows: list[ChoiceEvalResult] = []
    for qa, choice_letters, choice_logprobs, choice_probs in zip(
        qa_pairs,
        choice_letters_list,
        choice_logprobs_list,
        choice_probs_list,
        strict=True,
    ):
        rows.append(
            _build_choice_eval_result(
                persona,
                qa,
                condition,
                choice_letters=choice_letters,
                choice_logprobs=choice_logprobs,
                choice_probs=choice_probs,
                steering_layer=steering_layer,
                steering_alpha=steering_alpha,
            )
        )
    return rows
