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
    batch_size: int = 1,
    return_per_sample: bool = False,
) -> torch.Tensor:
    """Return mean hidden states with shape ``(num_layers, hidden_size)``.

    ``input_ids_list`` and ``token_masks`` must be aligned 1-D tensors. Passing
    pre-tokenized ids avoids retokenization and keeps masks aligned with the
    traced model input. Remote NDIF runs retry transient network failures.
    ``backend_factory`` can override backend construction for callers that need
    to bind per-request credentials while preserving fresh backends on retries.

    When ``batch_size > 1``, inputs are right-padded within each chunk and
    processed in a single forward pass per chunk. Padding tokens are excluded
    by per-sample token masks so masked means remain exact. Only correct for
    decoder-only models with causal attention (right-padding does not corrupt
    earlier token activations).

    When ``return_per_sample=True``, returns ``(n_inputs, num_layers, hidden_size)``
    instead of the cross-input mean ``(num_layers, hidden_size)``.
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

    # Padding is only needed by the batched path. Some local models do not
    # carry a tokenizer, so resolve a safe fallback only when it is needed.
    pad_id = 0
    if batch_size > 1:
        tokenizer = getattr(model, "tokenizer", None)
        pad_id = getattr(tokenizer, "pad_token_id", None) or 0

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

                for chunk_start in range(0, len(input_ids_list), batch_size):
                    chunk_ids = input_ids_list[chunk_start : chunk_start + batch_size]
                    chunk_masks = masks[chunk_start : chunk_start + batch_size]

                    if len(chunk_ids) == 1:
                        # Single-sample path: no padding overhead.
                        ids, mask = chunk_ids[0], chunk_masks[0]
                        hs: list[torch.Tensor] = []
                        # Pre-tokenized ids avoid double BOS insertion.
                        with model.trace(ids.unsqueeze(0)) as tracer:
                            for layer_idx in range(model.num_layers):
                                # (1, seq_len, hidden_size) → (seq_len, hidden_size)
                                layer_out = model.layers_output[layer_idx][0]
                                # mask → (num_masked, hidden_size) → mean: (hidden_size,)
                                layer_mean = layer_out[
                                    mask.to(device=layer_out.device)
                                ].mean(dim=0)
                                hs.append(layer_mean.detach().cpu())
                            per_text_hs = torch.stack(hs, dim=0)
                            # Extraction only needs residual activations; skipping the
                            # LM head avoids materializing full-sequence logits on NDIF.
                            tracer.stop()
                        per_sample_hs.append(per_text_hs)
                    else:
                        # Batched path: right-pad all inputs to max length in chunk.
                        max_len = max(ids.shape[0] for ids in chunk_ids)
                        padded = torch.stack(
                            [
                                torch.nn.functional.pad(
                                    ids, (0, max_len - ids.shape[0]), value=pad_id
                                )
                                for ids in chunk_ids
                            ]
                        )  # (chunk_size, max_len)
                        # Pad masks with False so they index into the padded sequence.
                        padded_masks = [
                            torch.cat(
                                [mask, mask.new_zeros(max_len - mask.shape[0], dtype=torch.bool)]
                            )
                            for mask in chunk_masks
                        ]

                        chunk_per_text_hs: list[list] = [[] for _ in range(len(chunk_ids))]
                        chunk_stacked: list = [None] * len(chunk_ids)
                        with model.trace(padded) as tracer:
                            for layer_idx in range(model.num_layers):
                                # (chunk_size, max_len, hidden_size)
                                layer_out = model.layers_output[layer_idx]
                                for j, pmask in enumerate(padded_masks):
                                    sample_out = layer_out[j]  # (max_len, hidden_size)
                                    sample_mean = sample_out[
                                        pmask.to(device=sample_out.device)
                                    ].mean(dim=0)
                                    chunk_per_text_hs[j].append(sample_mean.detach().cpu())
                            for j in range(len(chunk_ids)):
                                chunk_stacked[j] = torch.stack(chunk_per_text_hs[j], dim=0)
                            tracer.stop()
                        per_sample_hs.extend(chunk_stacked)

                stacked = torch.stack(per_sample_hs, dim=0)
                if return_per_sample:
                    result = stacked.save()
                else:
                    # (n_inputs, num_layers, hidden_size) → mean → (num_layers, hidden_size)
                    result = stacked.mean(dim=0).save()

            return result

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
