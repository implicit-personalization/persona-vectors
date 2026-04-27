"""Per-token activation extraction.

Parallel to ``extraction.py`` but saves the *full sequence* of hidden states
at each requested layer, rather than a masked mean. This is the input the
SAE pipeline wants — GemmaScope and similar SAEs are trained on per-token
residual-stream activations, so encoding token-by-token and pooling in
feature space (rather than encoding a pre-pooled vector) is the
distribution the SAE was trained on.

The averaged path remains for backwards compatibility; this module is
additive. Storage layout:

    artifacts/activations_pertoken/{model_dir}/{variant}/{persona_id}/
        layer_NN.pt    — list[Tensor], one per question, shape (seq_len_i, hidden)
        metadata.json  — questions + per-question token spans + saved layers
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import nnsight
import torch
from nnterp import StandardizedTransformer
from persona_data.synth_persona import PersonaData, QAPair

from persona_vectors.activations import _build_backend
from persona_vectors.artifacts import SUPPORTED_VARIANTS, PerTokenStore
from persona_vectors.extraction import prepare_variant_inputs


@dataclass
class PerTokenExtractionResult:
    variant: str
    output_dir: Path
    n_questions: int
    persona_name: str
    layers: list[int]


def extract_pertoken_activations(
    model: StandardizedTransformer,
    input_ids_list: list[torch.Tensor],
    layers: list[int],
    remote: bool = False,
    on_status: Callable[[str, str, str], None] | None = None,
) -> dict[int, list[torch.Tensor]]:
    """Run forward passes and return full per-token hidden states per layer.

    Args:
        model: Standardized nnterp model.
        input_ids_list: Pre-tokenized 1-D tensors, one per sample.
        layers: Layer indices to capture.
        remote: Run on NDIF remote servers.
        on_status: Optional NDIF status callback.

    Returns:
        ``{layer_idx: [Tensor(seq_len_i, hidden_size), ...]}`` — variable
        seq_len per sample.
    """
    if not layers:
        raise ValueError("layers must be a non-empty list")
    for ids in input_ids_list:
        if ids.ndim != 1:
            raise ValueError(f"expected 1-D input ids, got shape {tuple(ids.shape)}")

    backend = _build_backend(model, remote, on_status)

    # Mirror the pattern in ``extract_activations``: collect each input's
    # per-layer stack via ``nnsight.save``, then reshape after the session
    # closes. Avoids relying on ``.value`` which behaves differently locally
    # vs. on the remote NDIF backend.
    with torch.no_grad(), model.session(remote=remote, backend=backend):
        all_per_input: list[torch.Tensor] = nnsight.save([])

        for ids in input_ids_list:
            with model.trace(ids.unsqueeze(0)) as tracer:
                # All requested layers share seq_len for one input — stack
                # along a new axis so we save a single tensor per input.
                stacked = torch.stack(
                    [model.layers_output[layer][0].detach().cpu() for layer in layers],
                    dim=0,
                )
                per_input = nnsight.save(stacked)
                tracer.stop()

            all_per_input.append(per_input)

    per_layer: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    for stacked in all_per_input:
        # stacked shape: (n_layers_saved, seq_len_i, hidden_size)
        for i, layer in enumerate(layers):
            per_layer[layer].append(stacked[i])

    return per_layer


def run_pertoken_extraction(
    model: StandardizedTransformer,
    model_name: str,
    persona: PersonaData,
    qa_pairs: list[QAPair],
    variants: tuple[str, ...],
    layers: list[int],
    remote: bool = False,
    on_status: Callable | None = None,
    activations_dir: str | Path | None = None,
) -> list[PerTokenExtractionResult]:
    """Extract per-token activations and save them with span metadata.

    The per-token tensors are saved without averaging. We still reuse the
    normal prompt formatting path so the saved spans match the rest of the
    codebase.

    Args:
        model: Loaded nnterp model.
        model_name: HF identifier for artifact path.
        persona: Persona being extracted.
        qa_pairs: QA pairs to run.
        variants: Prompt variants ("templated", "biography", ...).
        layers: Layer indices to save. One per layer file.
        remote: Run on NDIF.
        on_status: NDIF status callback.
        activations_dir: Override storage root.

    Returns:
        One :class:`PerTokenExtractionResult` per variant.
    """
    if not qa_pairs:
        raise ValueError("No QA pairs selected for extraction")
    if invalid := set(variants) - set(SUPPORTED_VARIANTS):
        raise ValueError(f"Unsupported variants: {invalid}")

    store = PerTokenStore(model_name, root_dir=activations_dir)
    results: list[PerTokenExtractionResult] = []

    for variant in variants:
        prepared = prepare_variant_inputs(
            tokenizer=model.tokenizer,
            persona=persona,
            qa_pairs=qa_pairs,
            variant=variant,
        )

        per_layer = extract_pertoken_activations(
            model,
            input_ids_list=[p.input_ids for p in prepared],
            layers=layers,
            remote=remote,
            on_status=on_status,
        )

        artifact_dir = store.save(
            prompt_variant=variant,
            persona_id=persona.id,
            persona_name=persona.name,
            per_layer_tokens=per_layer,
            prepared=prepared,
        )

        results.append(
            PerTokenExtractionResult(
                variant=variant,
                output_dir=artifact_dir,
                n_questions=len(prepared),
                persona_name=persona.name,
                layers=list(layers),
            )
        )

    return results
