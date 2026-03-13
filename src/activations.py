import nnsight
import torch


def _get_hidden_states(layer_output):
    """Return the hidden-state tensor from a layer output."""
    if isinstance(layer_output, tuple):
        return layer_output[0]
    return layer_output


def extract_activations(
    model,
    full_texts: list[str],
    token_masks: list[torch.Tensor],
    remote: bool = False,
) -> list[torch.Tensor]:
    """Run forward passes and return masked mean activations per layer.

    Args:
        model: The nnsight LanguageModel.
        full_texts: List of full formatted prompt+response strings, one per sample.
        token_masks: List of boolean masks over the full token sequence per sample.
            True values are averaged.
        remote: If True, execute the trace on NDIF's remote servers instead of locally.
            Requires NDIF_API_KEY to be set (via .env or environment variable).
            When using remote=True, instantiate the model without device_map/dtype so
            it loads on the meta device (no local GPU needed).

    Returns:
        List of tensors with shape (n_layers, d_model), one per input text.
    """
    if len(full_texts) != len(token_masks):
        raise ValueError("full_texts and token_masks must have the same length")

    masks = [torch.as_tensor(mask, dtype=torch.bool) for mask in token_masks]
    if not all(mask.any() for mask in masks):
        raise ValueError("token_mask selects zero tokens")

    with model.session(remote=remote):
        all_hs: list[torch.Tensor] = nnsight.save([])

        for text, mask in zip(full_texts, masks):
            with model.trace(text):
                saved_hs = nnsight.save([])
                for layer in model.model.layers:
                    # WARNING: If this raises a RemoteException: RecursionError, the NDIF server is running
                    # nnsight <0.6.2 which has a ModuleList integer-index proxy bug. Wait for the server
                    # to update before upgrading to the latest version of nnsight
                    hidden_states = _get_hidden_states(layer.output)

                    # Compute the masked mean inside the trace so only (d_model) per layer
                    # is saved and transferred instead of the full (seq_len, d_model).
                    # Batch is intentionally always one here.
                    mask = mask.to(device=hidden_states.device)
                    layer_mean = (
                        hidden_states[:, mask, :].squeeze(dim=0).float().mean(dim=0)
                    )
                    saved_hs.append(layer_mean.detach().cpu().save())

            all_hs.append(torch.stack(list(saved_hs), dim=0))

    return list(all_hs)
