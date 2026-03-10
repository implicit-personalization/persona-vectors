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
    remote: bool = False,
) -> list[torch.Tensor]:
    """Run forward passes and return full hidden states per layer for each text.

    Args:
        model: The nnsight LanguageModel.
        full_texts: List of full formatted prompt+response strings, one per sample.
        remote: If True, execute the trace on NDIF's remote servers instead of locally.
            Requires NDIF_API_KEY to be set (via .env or environment variable).
            When using remote=True, instantiate the model without device_map/dtype so
            it loads on the meta device (no local GPU needed).

    Returns:
        List of tensors with shape (n_layers, seq_len, d_model), one per input text.
    """
    with model.session(remote=remote):
        all_hs: list[torch.Tensor] = nnsight.save([])

        for text in full_texts:
            with model.trace(text):
                saved_hs = nnsight.save([])
                for layer in model.model.layers:
                    # WARNING: If this raises a RemoteException: RecursionError, the NDIF server is running
                    # nnsight <0.6.2 which has a ModuleList integer-index proxy bug. Wait for the server
                    # to update before upgrading to the latest version of nnsight
                    hidden_states = _get_hidden_states(layer.output)

                    # Save the full sequence from (batch, seq_len, d_model) -> (seq_len, d_model)
                    # with batch intentionally always equal to one here.
                    layer_hs = hidden_states.squeeze(dim=0)
                    saved_hs.append(layer_hs.detach().cpu().save())

            all_hs.append(torch.stack(list(saved_hs), dim=0))

    return list(all_hs)
