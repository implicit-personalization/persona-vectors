import nnsight
import torch


def _get_hidden_states(layer_output):
    """Return the hidden-state tensor from a layer output."""
    if isinstance(layer_output, tuple):
        return layer_output[0]
    return layer_output


def _get_last_prompt_token_index(mask: torch.Tensor) -> int:
    """Infer prompt-final token index from a response-token mask."""
    response_positions = torch.nonzero(mask, as_tuple=False).flatten()
    if response_positions.numel() == 0:
        raise ValueError("token_mask selects zero tokens")

    response_start = int(response_positions[0].item())
    if response_start <= 0:
        raise ValueError("cannot infer last prompt token when response starts at index 0")

    return response_start - 1


def extract_activation_summaries(
    model,
    full_texts: list[str],
    token_masks: list[torch.Tensor],
    remote: bool = False,
) -> dict[str, torch.Tensor]:
    """Run forward passes and return response/prompt summaries per layer.

    Returns:
        Dict with:
            - response_mean: (n_text, n_layers, d_model)
            - last_prompt_token: (n_text, n_layers, d_model)
    """
    if len(full_texts) != len(token_masks):
        raise ValueError("full_texts and token_masks must have the same length")

    masks = [torch.as_tensor(m, dtype=torch.bool) for m in token_masks]
    last_prompt_indices = [_get_last_prompt_token_index(m) for m in masks]

    with model.session(remote=remote):
        all_response_means: list[torch.Tensor] = nnsight.save([])
        all_last_prompt_tokens: list[torch.Tensor] = nnsight.save([])

        for text, mask, prompt_idx in zip(full_texts, masks, last_prompt_indices):
            with model.trace(text):
                saved_response_means = nnsight.save([])
                saved_last_prompt_tokens = nnsight.save([])
                for layer in model.model.layers:
                    hidden_states = _get_hidden_states(layer.output)
                    mask_on_device = mask.to(device=hidden_states.device)

                    response_mean = (
                        hidden_states[:, mask_on_device, :].squeeze(dim=0).mean(dim=0)
                    )
                    last_prompt_token = hidden_states[:, prompt_idx, :].squeeze(dim=0)

                    saved_response_means.append(response_mean.detach().cpu().save())
                    saved_last_prompt_tokens.append(last_prompt_token.detach().cpu().save())

            all_response_means.append(torch.stack(list(saved_response_means), dim=0))
            all_last_prompt_tokens.append(torch.stack(list(saved_last_prompt_tokens), dim=0))

    return {
        "response_mean": torch.stack(all_response_means, dim=0),
        "last_prompt_token": torch.stack(all_last_prompt_tokens, dim=0),
    }


def extract_activations(
    model,
    full_texts: list[str],
    token_masks: list[torch.Tensor],
    remote: bool = False,
) -> torch.Tensor:
    """Run a forward passes and return mean hidden states over masked tokens per layer.

    Args:
        model: The nnsight LanguageModel.
        full_texts: List of full formatted prompt+response strings, one per sample.
        token_masks: List of boolean masks over the full (unpadded) token sequence per
            sample. True values are averaged. Each mask should match the length of the
            tokenized full_texts[i] without padding.
        remote: If True, execute the trace on NDIF's remote servers instead of locally.
            Requires NDIF_API_KEY to be set (via .env or environment variable).
            When using remote=True, instantiate the model without device_map/dtype so
            it loads on the meta device (no local GPU needed).
    """
    summaries = extract_activation_summaries(
        model=model,
        full_texts=full_texts,
        token_masks=token_masks,
        remote=remote,
    )
    return summaries["response_mean"]
