from typing import Callable

import nnsight
import torch
from nnsight.intervention.backends.remote import (
    JobStatusDisplay,
    RemoteBackend,
    RemoteException,
)
from nnterp import StandardizedTransformer


class _CallbackJobStatusDisplay(JobStatusDisplay):
    """JobStatusDisplay subclass that fires an extra callback on each update.

    The callback receives (job_id, status_name, description) so callers (e.g.
    a Streamlit UI) can render the NDIF job status without capturing stdout.
    """

    def __init__(self, on_status: Callable[[str, str, str], None], **kwargs):
        super().__init__(**kwargs)
        self._on_status = on_status

    def update(self, job_id: str = "", status_name: str = "", description: str = ""):
        super().update(job_id, status_name, description)
        if status_name:
            self._on_status(job_id, status_name, description)


def _build_backend(
    model: StandardizedTransformer,
    remote: bool,
    on_status: Callable[[str, str, str], None] | None,
):
    if not remote:
        return None

    backend = RemoteBackend(model.to_model_key())
    backend.CONNECT_TIMEOUT = 300.0
    if on_status is not None:
        backend.status_display = _CallbackJobStatusDisplay(
            on_status, enabled=True, verbose=backend.verbose
        )
    return backend


def extract_activations(
    model: StandardizedTransformer,
    input_ids_list: list[torch.Tensor],
    token_masks: list[torch.Tensor],
    remote: bool = False,
    on_status: Callable[[str, str, str], None] | None = None,
) -> torch.Tensor:
    """Run forward passes and return mean hidden states over masked tokens per layer.

    On remote (NDIF) execution, if the server raises an OOM the fast path is
    transparently retried with a layer-sliced trace that bounds peak memory.
    Local OOMs propagate.

    Args:
        model: The standardized nnterp model.
        input_ids_list: Pre-tokenized input id tensors, one 1-D tensor per sample.
            Passing ids (rather than strings) avoids any re-tokenization by
            nnsight so the masks always line up with what the model actually
            sees on the forward pass.
        token_masks: List of boolean masks over each sample's token sequence.
            True values are averaged. ``token_masks[i]`` must match the length
            of ``input_ids_list[i]``.
        remote: If True, execute the trace on NDIF's remote servers instead of locally.
            Requires NDIF_API_KEY to be set (via .env or environment variable).
        on_status: Optional callback ``(job_id, status_name, description) -> None``
            called on every NDIF status update. Only used when remote=True. Useful
            for surfacing job progress in a UI (e.g. Streamlit) without capturing
            stdout. The normal terminal output from nnsight is still printed.
    """

    if len(input_ids_list) != len(token_masks):
        raise ValueError("input_ids_list and token_masks must have the same length")

    masks = [torch.as_tensor(m, dtype=torch.bool) for m in token_masks]
    if not all(m.any() for m in masks):
        raise ValueError("token_mask selects zero tokens")
    for ids, mask in zip(input_ids_list, masks):
        if ids.ndim != 1:
            raise ValueError(f"expected 1-D input ids, got shape {tuple(ids.shape)}")
        if ids.shape[0] != mask.shape[0]:
            raise ValueError(
                f"input ids length {ids.shape[0]} does not match mask length {mask.shape[0]}"
            )

    try:
        return _extract_single_trace(model, input_ids_list, masks, remote, on_status)
    except RemoteException as e:
        if not remote or "OutOfMemoryError" not in e.tb_string:
            raise
        chunk_size = max(1, model.num_layers // 4)
        print(
            f"NDIF OOM on single-trace path; retrying with chunked extraction "
            f"(chunk_size={chunk_size})."
        )
        return _extract_chunked(
            model, input_ids_list, masks, remote, on_status, chunk_size=chunk_size
        )


def _extract_single_trace(
    model: StandardizedTransformer,
    input_ids_list: list[torch.Tensor],
    masks: list[torch.Tensor],
    remote: bool,
    on_status: Callable[[str, str, str], None] | None,
) -> torch.Tensor:
    backend = _build_backend(model, remote, on_status)
    with torch.no_grad(), model.session(remote=remote, backend=backend):
        all_hs: list[torch.Tensor] = nnsight.save([])

        for ids, mask in zip(input_ids_list, masks):
            saved_hs: list[torch.Tensor] = []
            # Pre-tokenized ids are passed straight through — no double-BOS.
            with model.trace(ids.unsqueeze(0)) as tracer:
                # All layers live on the same device, so move the mask once.
                mask_on_device = mask.to(device=model.layers_output[0].device)
                for layer_idx in range(model.num_layers):
                    # Extract activations: (1, seq_len, hidden_size):
                    #   remove batch dim: (seq_len, hidden_size)
                    #   mask them: (num_masked, hidden_size)
                    #   mean pool: (hidden_size,)
                    layer_mean = model.layers_output[layer_idx][0, mask_on_device].mean(
                        dim=0
                    )
                    saved_hs.append(layer_mean.detach().cpu())

                per_text_hs = nnsight.save(torch.stack(saved_hs, dim=0))
                # Extraction only needs residual activations; skipping the
                # LM head avoids materializing full-sequence logits on NDIF.
                tracer.stop()

            all_hs.append(per_text_hs)

    # Shape: (n_text, num_layers, hidden_size)
    return torch.stack(all_hs, dim=0)


def _extract_chunked(
    model: StandardizedTransformer,
    input_ids_list: list[torch.Tensor],
    masks: list[torch.Tensor],
    remote: bool,
    on_status: Callable[[str, str, str], None] | None,
    chunk_size: int,
) -> torch.Tensor:
    """Slice each forward pass across layers, carrying the boundary residual
    forward via ``model.skip_layers``. Slower (one NDIF round-trip per chunk)
    but bounds peak memory so long biographies fit."""
    all_hs: list[torch.Tensor] = []
    with torch.no_grad():
        for ids, mask in zip(input_ids_list, masks):
            chunk_hs: list[torch.Tensor] = []
            boundary: torch.Tensor | None = None
            # Pre-tokenized ids are passed straight through — no double-BOS.
            for start_layer in range(0, model.num_layers, chunk_size):
                end_layer = min(start_layer + chunk_size - 1, model.num_layers - 1)
                chunk_backend = _build_backend(model, remote, on_status)
                with model.session(remote=remote, backend=chunk_backend):
                    with model.trace(ids.unsqueeze(0)) as tracer:
                        # All layers in a trace live on the same device.
                        layer_device = (
                            torch.device("cuda") if remote else model.layers[0].device
                        )
                        if boundary is not None:
                            model.skip_layers(
                                0, start_layer - 1, skip_with=boundary.to(layer_device)
                            )

                        mask_on_device = mask.to(device=layer_device)
                        saved_hs = []
                        for layer_idx in range(start_layer, end_layer + 1):
                            layer_mean = model.layers_output[layer_idx][
                                0, mask_on_device
                            ].mean(dim=0)
                            saved_hs.append(layer_mean.detach().cpu())

                        per_chunk_hs = nnsight.save(torch.stack(saved_hs, dim=0))
                        next_boundary = (
                            model.layers_output[end_layer].save()
                            if end_layer < model.num_layers - 1
                            else None
                        )
                        tracer.stop()

                chunk_hs.append(per_chunk_hs)
                boundary = next_boundary

            all_hs.append(torch.cat(chunk_hs, dim=0))

    # Shape: (n_text, num_layers, hidden_size)
    return torch.stack(all_hs, dim=0)
