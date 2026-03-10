# HACK: This file is planning to add support also for system token and for returning additional informations on the indixies in the tokens
# It is still in the works and should be reviewd more carefully and changed accordingly to what we want to do


def _supports_system_role(tokenizer) -> bool:
    """Check if tokenizer's chat template supports the 'system' role."""
    # HACK: We need to test out this is actually working as expected with differenet models !
    # This comment can be removed after things have been tested out
    try:
        tokenizer.apply_chat_template(
            [{"role": "system", "content": "test"}],
            tokenize=False,
        )
        return True
    except Exception:
        return False


def _normalize_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Merge any leading system message into the first user message.

    Only applies the merge if the tokenizer's chat template doesn't support
    the "system" role (e.g., Gemma 2). Otherwise leaves messages unchanged.
    """
    if not messages or messages[0]["role"] != "system":
        return messages
    system_content = messages[0]["content"]
    rest = list(messages[1:])
    if rest and rest[0]["role"] == "user" and system_content:
        rest[0] = {
            "role": "user",
            "content": f"{system_content}\n\n{rest[0]['content']}",
        }
    return rest


def format_messages(messages: list[dict[str, str]], tokenizer) -> tuple[str, int]:
    """Format a conversation for the model using its chat template.

    Args:
        messages: List of message dicts with "role" and "content" keys.
                 Can include "system", "user", and "assistant" roles.
        tokenizer: The tokenizer with chat template support.

    Returns:
        full_prompt: The full formatted prompt as a string.
        response_start_idx: The token index of the first token in the last
                            assistant message.
    """
    # HACK: Exact question/context token boundaries depend on the model's chat
    # template. For now we keep this simple and only compute the assistant span,
    # which is the only boundary used downstream. Revisit when we want a differnet span of tokens
    supports_system = _supports_system_role(tokenizer)
    if not supports_system:
        messages = _normalize_messages(messages)

    full_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )

    if len(messages) <= 1:
        prompt_without_response = full_prompt
    else:
        prompt_without_response = tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True
        )

    prompt_input_ids = tokenizer(prompt_without_response, return_tensors="pt").input_ids
    response_start_idx = prompt_input_ids.shape[1]

    return full_prompt, response_start_idx
