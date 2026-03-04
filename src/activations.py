import torch


def extract_activations(
    model,
    full_text: str,
    token_mask: list[bool] | torch.Tensor,
) -> list[torch.Tensor]:
    """Run a single forward pass and return mean hidden states over masked tokens per layer.

    Args:
        model: The nnsight LanguageModel.
        full_text: The full formatted prompt+response string.
        token_mask: Boolean mask over the full token sequence. True values are averaged.

    Notes:
        - For multi-turn conversations, build a mask that selects the tokens you
          want to average over.
    """
    saved_hs = []

    # HACK: Possible Simple trace to get the activations.
    # It would be nice to be able to run with VLLM here and also remotely.
    with model.trace(full_text):
        for layer in model.model.layers:
            # layer.output is the hidden state tensor of shape (batch, seq_len, d_model).
            # Gemma-2's decoder layer returns a plain tensor, not a tuple.
            # NOTE: This is architecture-specific — verify the correct attribute if switching models.
            saved_hs.append(layer.output.save())

    mask = torch.as_tensor(token_mask, dtype=torch.bool)
    if not mask.any():
        raise ValueError("token_mask selects zero tokens")

    # Mean activations only over only the masked tokens
    return [s[:, mask, :].squeeze(0).float().mean(dim=0).cpu() for s in saved_hs]
