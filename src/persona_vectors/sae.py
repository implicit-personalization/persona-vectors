"""SAE loading and feature-space persona analysis.

Loads any JumpReLU SAE published as ``.safetensors`` or ``.npz`` on the HF
Hub. Covers GemmaScope (``google/gemma-scope-*``), LlamaScope, and any SAE
using the convention:

    W_enc:     (d_model, d_sae)
    W_dec:     (d_sae,   d_model)
    b_enc:     (d_sae,)
    b_dec:     (d_model,)
    threshold: (d_sae,)         # JumpReLU gate

The intended pipeline (matching Arad et al. 2025 / io-analysis):

    per-token activations  →  sae.encode  →  pool over response tokens
                                       (mean in feature space)
                                  ↓
    feature_vec(biography)  −  feature_vec(templated)
                                  ↓
                          top-K differential features

This is the *correct* distribution to encode through the SAE — encoding a
pre-pooled vector projects off-manifold and produces poor reconstructions
(see io-analysis README).

For steering at generation time, ``error_preserving_steer`` follows the
nnsight-vllm-demos pattern: features the SAE doesn't capture pass through
unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


class JumpReLUSAE(nn.Module):
    """JumpReLU sparse autoencoder.

    ``encode(x) = JumpReLU(W_enc·(x − b_dec) + b_enc; threshold)``
    ``decode(a) = a·W_dec + b_dec``

    Inputs and outputs are ``(..., d_model)``; activations are ``(..., d_sae)``.
    """

    def __init__(self, d_model: int, d_sae: int):
        super().__init__()
        self.W_enc = nn.Parameter(torch.zeros(d_model, d_sae))
        self.W_dec = nn.Parameter(torch.zeros(d_sae, d_model))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        self.threshold = nn.Parameter(torch.zeros(d_sae))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = (x - self.b_dec) @ self.W_enc + self.b_enc
        return torch.where(pre > self.threshold, pre, torch.zeros_like(pre))

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        return acts @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    @torch.no_grad()
    def error_preserving_steer(
        self,
        x: torch.Tensor,
        modifications: list[tuple[int, float]],
    ) -> torch.Tensor:
        """Encode → modify features → decode + reconstruction error.

        Pattern from ndif-team/nnsight-vllm-demos: the features the SAE
        doesn't capture survive in ``error = x − decode(encode(x))`` and
        are added back to the modified reconstruction. Without this, the
        intervention destroys whatever the SAE failed to model.

        Args:
            x: ``(..., d_model)`` activations.
            modifications: ``(feature_index, delta)`` pairs. Positive values
                add to the feature, negative values subtract, and ``0`` zeros
                the feature.

        Returns:
            Modified activations, same shape as ``x``.
        """
        encoded = self.encode(x)
        recon = self.decode(encoded)
        error = x - recon
        for idx, scale in modifications:
            if scale != 0:
                encoded[..., idx] += scale
            else:
                encoded[..., idx] = 0
        return self.decode(encoded) + error


@dataclass
class SAEHandle:
    """A loaded SAE plus the HF coordinates it came from."""

    sae: JumpReLUSAE
    repo_id: str
    filename: str
    d_model: int
    d_sae: int


def load_jumprelu_sae(repo_id: str, filename: str) -> SAEHandle:
    """Download a JumpReLU SAE checkpoint from HF and return a wrapped module.

    Args:
        repo_id: HuggingFace repo (e.g. ``"google/gemma-scope-9b-it-res"``).
        filename: Path inside the repo. For GemmaScope this looks like
            ``"layer_20/width_16k/average_l0_91/params.npz"``.
    """
    local_path = hf_hub_download(repo_id, filename)

    if local_path.endswith(".npz"):
        import numpy as np

        npz = np.load(local_path)
        params = {k: torch.from_numpy(npz[k]) for k in npz.files}
    else:
        params = load_file(local_path)

    if "W_enc" not in params or "W_dec" not in params:
        raise KeyError(
            f"SAE checkpoint {repo_id}/{filename} is missing W_enc/W_dec — "
            f"got keys: {sorted(params)}"
        )

    d_model, d_sae = params["W_enc"].shape
    sae = JumpReLUSAE(d_model=d_model, d_sae=d_sae)
    expected = set(dict(sae.named_parameters()))
    sae.load_state_dict({k: v.float() for k, v in params.items() if k in expected})
    sae.eval()

    return SAEHandle(
        sae=sae, repo_id=repo_id, filename=filename, d_model=d_model, d_sae=d_sae
    )


def encode_response_features(
    tokens: torch.Tensor,
    response_span: tuple[int, int],
    sae: JumpReLUSAE,
) -> torch.Tensor:
    """SAE-encode a token sequence and mean-pool over the response span.

    This is the in-distribution operation for a per-token SAE: encode each
    token, then pool in *feature* space, never in residual space.

    Args:
        tokens: ``(seq_len, d_model)`` per-token residual stream.
        response_span: ``(start, end)`` token indices to pool over (typically
            the assistant response).
        sae: Loaded JumpReLU SAE.

    Returns:
        ``(d_sae,)`` mean feature activation over the response tokens.
    """
    start, end = response_span
    if end <= start or end > tokens.shape[0]:
        raise ValueError(
            f"invalid response_span {response_span} for seq_len {tokens.shape[0]}"
        )
    response_tokens = tokens[start:end].to(sae.W_enc.dtype)
    with torch.no_grad():
        feats = sae.encode(response_tokens)  # (n_response, d_sae)
    return feats.mean(dim=0)


def feature_space_steering_vector(
    pos_features: torch.Tensor,
    neg_features: torch.Tensor,
    top_k: int = 20,
) -> dict:
    """Mean-diff in feature space, return top-K differential features.

    Args:
        pos_features: ``(n_questions_pos, d_sae)`` — feature vectors for the
            positive variant (e.g. biography), one per question.
        neg_features: ``(n_questions_neg, d_sae)`` — same for negative
            (e.g. templated).
        top_k: Number of top differential features to return.

    Returns:
        Dict with:
          ``feature_diff``: ``(d_sae,)`` — ``mean(pos) − mean(neg)``.
          ``feature_ids``: ``(top_k,)`` long — indices ranked by ``|diff|``.
          ``activations``: ``(top_k,)`` — signed diff values for those ids.
          ``mean_pos`` / ``mean_neg``: ``(d_sae,)`` per-feature means.
    """
    if pos_features.shape[-1] != neg_features.shape[-1]:
        raise ValueError("pos/neg feature vectors have different d_sae")

    mean_pos = pos_features.mean(dim=0)
    mean_neg = neg_features.mean(dim=0)
    diff = mean_pos - mean_neg

    top = diff.abs().topk(min(top_k, diff.numel()))
    return {
        "feature_diff": diff,
        "feature_ids": top.indices,
        "activations": diff[top.indices],
        "mean_pos": mean_pos,
        "mean_neg": mean_neg,
    }


def decode_to_residual(
    feature_diff: torch.Tensor,
    sae: JumpReLUSAE,
    feature_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Project a feature-space vector back into residual stream space.

    Use this to turn an SAE-derived persona direction into a steering vector
    consumable by the existing ``alpha · sv`` hook.

    Args:
        feature_diff: ``(d_sae,)`` — feature-space coefficients to decode.
        sae: Loaded SAE.
        feature_ids: Optional subset of feature indices to keep. If given,
            all other features are zeroed before decoding.

    Returns:
        ``(d_model,)`` residual-stream vector. Note: ``b_dec`` is *not* added
        — this is a direction, not an absolute residual.
    """
    coeffs = feature_diff.clone()
    if feature_ids is not None:
        mask = torch.zeros_like(coeffs)
        mask[feature_ids] = 1.0
        coeffs = coeffs * mask
    with torch.no_grad():
        return coeffs @ sae.W_dec  # (d_model,) — decoder bias deliberately excluded


def logit_lens(
    sae: JumpReLUSAE,
    norm_weight: torch.Tensor,
    W_unembed: torch.Tensor,
    feature_ids: torch.Tensor | None = None,
    k: int = 10,
) -> torch.Tensor:
    """Project SAE decoder columns through the model unembedding (Arad Eq. 4).

    Args:
        sae: Loaded SAE.
        norm_weight: Final RMSNorm scale, shape ``(d_model,)``.
        W_unembed: Unembedding matrix, shape ``(vocab_size, d_model)``.
        feature_ids: Specific feature indices to lens; ``None`` = all.
        k: Top tokens per feature.

    Returns:
        ``(n_features, k)`` long tensor of vocabulary token ids.
    """
    W_dec = sae.W_dec.detach().float()
    if feature_ids is not None:
        W_dec = W_dec[feature_ids]

    rms = W_dec.pow(2).mean(dim=-1, keepdim=True).sqrt() + 1e-6
    normed = (W_dec / rms) * norm_weight.float().unsqueeze(0)
    logits = normed @ W_unembed.float().T
    return logits.topk(k=k, dim=-1).indices


def neuronpedia_url(
    feature_id: int,
    sae_release: str = "gemma-2-9b-it",
    layer: int = 20,
    width: str = "16k",
) -> str:
    """Build a Neuronpedia URL for a feature.

    Layout: ``https://www.neuronpedia.org/{model}/{layer}-gemmascope-res-{width}/{feature}``.
    Pretrained 2B SAEs are indexed under ``gemma-2-2b`` (no ``-it`` suffix).
    """
    return (
        f"https://www.neuronpedia.org/{sae_release}/"
        f"{layer}-gemmascope-res-{width}/{feature_id}"
    )
