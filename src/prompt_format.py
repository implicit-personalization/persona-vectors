def format_biography_prompt(biography_md: str) -> str:
    _, _, biography_body = biography_md.partition("\n")
    biography_body = biography_body.lstrip()

    prompt = f"""You are roleplaying as a specific person in a conversation.
Stay fully in character. Be truthful to the profile below.
Do not mention that you are an AI model.

# Person biography:

{biography_body}

ROLEPLAY GUIDELINES:

- Answer naturally and conversationally as this person."""
    return prompt


def format_templated_prompt(templated_prompt: str) -> str:
    TEMPLATED_PROMPT_STRIP_LINE = "- If a question asks for details not supported by the profile, respond with plausible uncertainty or say you don't know, rather than inventing facts."

    lines = [
        line
        for line in templated_prompt.splitlines()
        if line.strip() != TEMPLATED_PROMPT_STRIP_LINE
    ]
    return "\n".join(lines).strip()


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
