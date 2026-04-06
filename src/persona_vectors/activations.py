import nnsight
import torch
from nnterp import StandardizedTransformer


def extract_activations(
    model: StandardizedTransformer,
    full_texts: list[str],
    token_masks: list[torch.Tensor],
    remote: bool = False,
) -> torch.Tensor:
    """Run forward passes and return mean hidden states over masked tokens per layer.

    Args:
        model: The standardized nnterp model.
        full_texts: List of full formatted prompt+response strings, one per sample.
        token_masks: List of boolean masks over the full (unpadded) token sequence per
            sample. True values are averaged. Each mask should match the length of the
            tokenized full_texts[i] without padding.
        remote: If True, execute the trace on NDIF's remote servers instead of locally.
            Requires NDIF_API_KEY to be set (via .env or environment variable).
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
                # Compute the masked mean inside the trace so only (num_layers, hidden_size)
                with model.trace(text):
                    for layer_idx in range(model.num_layers):
                        hidden_states = model.layers_output[layer_idx]
                        mask_on_device = mask.to(device=hidden_states.device)
                        # Take the mean over the masked tokens
                        # from (batch, seq_len, hidden_size) -> with batch = 1
                        # -> stripping batch dimension which intentionally will always be one
                        layer_mean = (
                            hidden_states[:, mask_on_device, :]
                            .squeeze(dim=0)
                            .mean(dim=0)
                        )
                        saved_hs.append(layer_mean.detach().cpu())

                    per_text_hs = nnsight.save(torch.stack(saved_hs, dim=0))

                all_hs.append(per_text_hs)

    # Shape: (n_text, num_layers, hidden_size)
    return torch.stack(all_hs, dim=0)
