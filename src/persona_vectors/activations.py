import time
from typing import Callable

import httpx
import socketio.exceptions
import torch
from nnsight.intervention.backends.remote import JobStatusDisplay, RemoteBackend
from nnterp import StandardizedTransformer

NETWORK_ERRORS = (
    socketio.exceptions.ConnectionError,
    TimeoutError,
    httpx.TransportError,
)


class CallbackJobStatusDisplay(JobStatusDisplay):
    """JobStatusDisplay subclass that fires an extra callback on each update.

    The callback receives ``(job_id, status_name, description)`` so callers
    (e.g. a Streamlit UI) can render NDIF job status without capturing stdout.
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
        backend.status_display = CallbackJobStatusDisplay(
            on_status, enabled=True, verbose=backend.verbose
        )
    return backend


def extract_activations(
    model: StandardizedTransformer,
    input_ids_list: list[torch.Tensor],
    token_masks: list[torch.Tensor],
    remote: bool = False,
    on_status: Callable[[str, str, str], None] | None = None,
    backend_factory: Callable[[], object] | None = None,
) -> torch.Tensor:
    """Return mean hidden states with shape ``(num_layers, hidden_size)``.

    ``input_ids_list`` and ``token_masks`` must be aligned 1-D tensors. Passing
    pre-tokenized ids avoids retokenization and keeps masks aligned with the
    traced model input. Remote NDIF runs retry transient network failures.
    ``backend_factory`` can override backend construction for callers that need
    to bind per-request credentials while preserving fresh backends on retries.
    """

    if len(input_ids_list) != len(token_masks):
        raise ValueError("input_ids_list and token_masks must have the same length")
    if not input_ids_list:
        raise ValueError("input_ids_list must contain at least one sample")

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

    # Remote sessions are lost on websocket or artifact-download failures.
    max_retries = 3 if remote else 1
    for attempt in range(max_retries):
        try:
            backend = (
                backend_factory()
                if backend_factory is not None
                else _build_backend(model, remote, on_status)
            )
            with torch.no_grad(), model.session(remote=remote, backend=backend):
                per_sample_hs: list[torch.Tensor] = []

                for ids, mask in zip(input_ids_list, masks):
                    hs: list[torch.Tensor] = []
                    # Pre-tokenized ids avoid double BOS insertion.
                    with model.trace(ids.unsqueeze(0)) as tracer:
                        for layer_idx in range(model.num_layers):
                            # Extract activations: (1, seq_len, hidden_size):
                            #   remove batch dim: (seq_len, hidden_size)
                            layer_out = model.layers_output[layer_idx][0]
                            # mask them: (num_masked, hidden_size) -> mean pool: (hidden_size,)
                            layer_mean = layer_out[
                                mask.to(device=layer_out.device)
                            ].mean(dim=0)
                            hs.append(layer_mean.detach().cpu())

                        per_text_hs = torch.stack(hs, dim=0)
                        # Extraction only needs residual activations; skipping the
                        # LM head avoids materializing full-sequence logits on NDIF.
                        tracer.stop()

                    per_sample_hs.append(per_text_hs)

                # (n_text, num_layers, hidden_size): average across text ->(num_layers, hidden_size)
                persona_vectors = torch.stack(per_sample_hs, dim=0).mean(dim=0).save()

            return persona_vectors

        except NETWORK_ERRORS as e:
            if attempt == max_retries - 1:
                raise
            wait = 30 * (2**attempt)  # 30s, 60s
            msg = (
                f"NDIF connection dropped ({type(e).__name__}); retrying in {wait}s "
                f"(attempt {attempt + 1}/{max_retries - 1})."
            )
            print(msg)
            if on_status is not None:
                on_status("local", "retry", msg)
            time.sleep(wait)
