def _normalize_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Merge a leading system message into the first user message."""
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
    normalized_messages = messages

    try:
        full_prompt = tokenizer.apply_chat_template(
            normalized_messages, tokenize=False, add_generation_prompt=False
        )
    except Exception:
        normalized_messages = _normalize_messages(messages)
        full_prompt = tokenizer.apply_chat_template(
            normalized_messages, tokenize=False, add_generation_prompt=False
        )

    if len(normalized_messages) <= 1:
        prompt_without_response = tokenizer.apply_chat_template(
            normalized_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    else:
        prompt_without_response = tokenizer.apply_chat_template(
            normalized_messages[:-1],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )

    response_start_idx = prompt_without_response["input_ids"].shape[1]

    return full_prompt, response_start_idx
