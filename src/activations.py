import nnsight
import torch


# HACK: Pehrpas there is a cleaner way to do this Idk to be honest
def extract_activations(
    model,
    full_texts: list[str],
    token_masks: list[torch.Tensor],
    remote: bool = False,
) -> torch.Tensor:
    """Run a single batched forward pass and return mean hidden states over masked tokens per layer.

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

    masks = [torch.as_tensor(m, dtype=torch.bool) for m in token_masks]
    if not all(m.any() for m in masks):
        raise ValueError("token_mask selects zero tokens")

    # Tokenize once with padding so we can reuse input_ids in the trace (no re-tokenization).
    # nnsight left-pads, so real tokens for sample i are right-aligned in the padded sequence.
    # lengths[i] = number of real tokens; pad count = max_len - lengths[i].
    encodings = model.tokenizer(full_texts, padding=True, return_tensors="pt")
    max_len = encodings.input_ids.shape[1]
    lengths = encodings.attention_mask.sum(dim=1)  # (batch,) real token counts

    # Right-align each unpadded mask by prepending False for the padding positions.
    # After this, padded_masks[i] is True exactly where we want to average over
    batch = len(masks)

    # (batch, max_len)
    padded_masks = torch.stack(
        [
            torch.cat([torch.zeros(max_len - lengths[i], dtype=torch.bool), masks[i]])
            for i in range(batch)
        ]
    )

    # Compute the masked mean inside the trace so only (batch, d_model) per layer is saved
    # and transferred, instead of the full (batch, max_len, d_model) activations.
    # mask_f is a plain tensor serialised with the request — it's tiny.
    with model.trace(encodings, remote=remote):
        saved_hs = nnsight.save([])
        for layer in model.model.layers:
            # layer.output: (batch, max_len, d_model)
            # Gemma's decoder layer returns a plain tensor, not a tuple.
            # NOTE: This is architecture-specific — verify the correct attribute if switching models.

            # WARNING: If this raises a RemoteException: RecursionError, the NDIF server is running
            # nnsight <0.6.2 which has a ModuleList integer-index proxy bug. Wait for the server
            # to update before upgrading to the latest version of nnsight

            layer_hs = []
            for i in range(batch):
                masked = layer.output[i][padded_masks[i]]
                layer_hs.append(masked.mean(dim=0).detach().cpu())

            # Stack over the batch -> (batch, d_model).
            saved_hs.append(torch.stack(layer_hs))

    # Shape: (batch, n_layers, d_model)
    return torch.stack(saved_hs, dim=1)
