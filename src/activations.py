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
) -> torch.Tensor:
    """Run forward passes and return mean hidden states over masked tokens per layer.

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

    if len(full_texts) != len(token_masks):
        raise ValueError("full_texts and token_masks must have the same length")

    masks = [torch.as_tensor(m, dtype=torch.bool) for m in token_masks]
    if not all(m.any() for m in masks):
        raise ValueError("token_mask selects zero tokens")

    with torch.no_grad():
        with model.session(remote=remote):
            all_hs: list[torch.Tensor] = nnsight.save([])

            for text, mask in zip(full_texts, masks):
                saved_hs: list[torch.Tensor] = []
                # Compute the masked mean inside the trace so only (n_layer ,d_model)
                with model.trace(text):
                    for layer in model.model.layers:
                        # WARNING: If this raises a RemoteException: RecursionError, the NDIF server is running
                        # nnsight <0.6.2 which has a ModuleList integer-index proxy bug. Wait for the server
                        # to update before upgrading to the latest version of nnsight

                        # Take the mean over the masked tokens
                        # from (batch, seq_len, d_model) -> with batch = 1
                        # -> stripping batch dimension which intentinally will always be one
                        hidden_states = _get_hidden_states(layer.output)
                        mask_on_device = mask.to(device=hidden_states.device)
                        layer_mean = (
                            hidden_states[:, mask_on_device, :]
                            .squeeze(dim=0)
                            .mean(dim=0)
                        )
                        saved_hs.append(layer_mean.detach().cpu())

                    per_text_hs = nnsight.save(torch.stack(saved_hs, dim=0))

                all_hs.append(per_text_hs)

    # Shape: (n_text, n_layers, d_model)
    return torch.stack(all_hs, dim=0)
