# NOTE: This can be changed to work with system prompt instead
# HACK: I also will review this more carefully if nobody else is going to do it when we have a real example of dataset


def _normalize_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Merge any leading system message into the first user message.

    Gemma 2's chat template raises an error for the "system" role. The standard
    workaround is to prepend the system content to the first user message.
    This normalization is applied unconditionally so the format is consistent
    across all models regardless of whether they support the system role.
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
                 Any leading system message is always merged into the first user
                 message so the format is consistent across all models (including
                 Gemma 2, which does not support the system role).
        tokenizer: The tokenizer with chat template support.

    Returns:
        full_prompt: The full formatted prompt as a string.
        response_start_idx: The token index of the first token in the last
                            assistant message.
    """
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

    response_start_idx = tokenizer(
        prompt_without_response, return_tensors="pt"
    ).input_ids.shape[1]

    return full_prompt, response_start_idx
