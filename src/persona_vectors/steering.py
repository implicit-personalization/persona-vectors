"""
Generate persona steering vectors from pre-extracted activations.

Method: Contrastive mean-diff
──────────────────────────────────────────────────────────────────────────────
For each persona:

  negative prompt  ->  Templated prompt + Question + Answer
  positive prompt  ->  Biography + Question + Answer

Load the saved mean response-token hidden states at STEER_LAYER. The activation
artifacts are already averaged across QA pairs and masked tokens, so each one
is a single (num_layers, hidden_size) tensor per (variant, persona).

  steering_vector = biography_h[layer] - templated_h[layer]

Output saved to: artifacts/vectors/{persona_id}/steering_vector.safetensors
"""

import json
from collections.abc import Sequence
from pathlib import Path

import torch
from persona_data.prompts import format_messages
from rich.console import Console
from rich.table import Table
from safetensors.torch import load_file, save_file

from persona_vectors.artifacts import PersonaVectorStore

# Config
STEER_LAYER = 20

console = Console()


def compute_steering_vector(
    persona_id: str,
    model_name: str,
    layer_idx: int = STEER_LAYER,
    mask_strategy: object | None = None,
    activations_dir: Path | str | None = None,
    verbose: bool = True,
) -> dict:
    """Load saved activations and compute steering vector.

    Uses pre-extracted masked-mean activations (response tokens only) from
    both biography and templated prompt variants.

    Args:
        persona_id: Persona UUID.
        model_name: HF model name (used to locate activation directory).
        layer_idx: Layer index for the steering vector.
        activations_dir: Root directory for activations. Defaults to
            artifacts/activations.
        verbose: If True, print progress and summary table.

    Returns:
        Dict with steering_vector, suggested_alpha, and metadata.
    """
    store = PersonaVectorStore(model_name, activations_dir)

    if verbose:
        print(f"\nLoading activations for {persona_id}...")

    # Load both positive (biography) and negative (templated) activation means.
    try:
        pos_activations = store.load(
            "biography", persona_id, mask_strategy=mask_strategy
        )
    except FileNotFoundError:
        if verbose:
            print("✗ Biography activations not found. Run extraction first.")
        return {}

    try:
        neg_activations = store.load(
            "templated", persona_id, mask_strategy=mask_strategy
        )
    except FileNotFoundError:
        if verbose:
            print("✗ Templated activations not found. Run extraction first.")
        return {}

    # Verify shape alignment.
    if pos_activations.shape != neg_activations.shape:
        raise ValueError(
            "Mismatch: "
            f"{tuple(pos_activations.shape)} positive vs "
            f"{tuple(neg_activations.shape)} negative"
        )
    if layer_idx < 0 or layer_idx >= pos_activations.shape[0]:
        raise ValueError(
            f"layer_idx {layer_idx} is out of range for {pos_activations.shape[0]} layers"
        )

    if verbose:
        print("Computing steering direction...")
    pos_mean = pos_activations[layer_idx, :].float()
    neg_mean = neg_activations[layer_idx, :].float()
    raw_sv = pos_mean - neg_mean

    # Steering vector shape: [1, 1, hidden_dim]
    steering_vector = raw_sv.unsqueeze(0).unsqueeze(0)

    # Calculate Alpha (20x Mean RMS of negatives)
    mean_rms = neg_mean.pow(2).mean().sqrt().item()
    sv_norm = raw_sv.norm().item()
    suggested_alpha = (20.0 * mean_rms) / (sv_norm + 1e-8)

    if verbose:
        table = Table(title="Steering Vector Summary")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_row("Persona ID", persona_id)
        table.add_row("Layer", str(layer_idx))
        table.add_row("Shape", str(tuple(steering_vector.shape)))
        table.add_row("L2 Norm", f"{sv_norm:.6f}")
        table.add_row("Mean Neg RMS", f"{mean_rms:.6f}")
        table.add_row("Suggested Alpha", f"{suggested_alpha:.4f}")
        table.add_row("Hidden Size", str(pos_activations.shape[1]))
        console.print(table)

    return {
        "steering_vector": steering_vector,
        "suggested_alpha": suggested_alpha,
        "persona_id": persona_id,
        "layer": layer_idx,
        "model_id": model_name,
        "hidden_size": int(pos_activations.shape[1]),
    }


def save_steering_vector(
    sv_dict: dict, out_path: str | Path, verbose: bool = True
) -> Path:
    """Save steering vector dict to safetensors + metadata.json.

    Args:
        sv_dict: Dict from compute_steering_vector.
        out_path: Directory to save into. Creates {persona_id}.safetensors
            and metadata.json inside.
        verbose: If True, print confirmation.

    Returns:
        Path to the output directory.
    """
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    # Save tensor as safetensors
    tensor_path = out_path / "steering_vector.safetensors"
    save_file(
        {"steering_vector": sv_dict["steering_vector"].detach().cpu()},
        str(tensor_path),
    )

    # Save metadata as JSON
    metadata = {
        "suggested_alpha": sv_dict["suggested_alpha"],
        "persona_id": sv_dict["persona_id"],
        "layer": sv_dict["layer"],
        "model_id": sv_dict["model_id"],
        "hidden_size": sv_dict["hidden_size"],
    }
    metadata_path = out_path / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    if verbose:
        print(f"\nSaved to {out_path}")
    return out_path


def load_steering_vector(path: str | Path) -> dict:
    """Load a saved steering vector from safetensors + metadata.json.

    Args:
        path: Directory containing steering_vector.safetensors and metadata.json.

    Returns:
        Dict with steering_vector tensor and metadata fields.
    """
    path = Path(path)

    tensors = load_file(str(path / "steering_vector.safetensors"))
    metadata = json.loads((path / "metadata.json").read_text())

    return {
        "steering_vector": tensors["steering_vector"],
        **metadata,
    }


# ── Causal steering during generation (nnterp / nnsight) ──────────────────────
#
#   with model.generate(prompt, remote=remote) as tracer:
#       with tracer.all():                      # apply to *every* decoded step
#           model.steer(layers=L,               # residual stream after block L
#                       steering_vector=v,       # 1-D tensor, hidden_size
#                       factor=c)                # adds c * v to that residual
#       out = model.generator.output.save()
#
# ``tracer.all()`` scopes the intervention to all generated positions (without
# it, only the prompt's last position is steered). ``model.steer`` registers an
# additive hook on the layer's residual output: h <- h + factor * vector.


def steering_coefficient(info: dict, strength: float, *, sign: float = 1.0) -> float:
    """Calibrated steering coefficient in *gap units* from a direction ``info``.

    ``info`` is a direction dict (e.g. from
    :func:`persona_vectors.traits.build_trait_direction`). ``strength`` is in gap
    units: ``strength=1`` lands the activation at the opposite-class centroid
    (``coefficient = strength * gap_norm``). Scaling by the residual *norm*
    (``act_norm``) instead over-steers 8-28x the real class separation and
    collapses the model off-manifold. ``sign`` flips the causal polarity: a
    decode-optimal difference-of-means axis can steer the *opposite* way
    behaviourally, so pass ``sign=-1`` to invert.
    """
    return float(strength * sign * info["gap_norm"])


def _format_chat(model, prompt: str, system: str | None) -> tuple[str, int]:
    """Apply the chat template to ``(system?, user)`` → ``(text, prompt_token_count)``.

    ``format_messages`` normalizes the system turn for models like gemma that lack
    one; the token count is used to slice the prompt off the decoded output.
    """
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    return format_messages(messages, model.tokenizer, add_generation_prompt=True)


def _decode_continuation(model, out, prompt_token_count: int) -> str:
    """Decode the continuation, slicing the prompt off by token count.

    Slicing by count (not string prefix) is required: ``skip_special_tokens`` drops
    the chat-template markers, so a string-prefix test would let the prompt leak in.
    """
    return model.tokenizer.decode(
        out[0][prompt_token_count:], skip_special_tokens=True
    ).strip()


def generate_steered_once(
    model,
    prompt: str,
    *,
    layer: int | None = None,
    steering_vector: torch.Tensor | None = None,
    factor: float = 0.0,
    remote: bool = True,
    backend=None,
    **generation_kwargs,
) -> torch.Tensor:
    """One generation pass, optionally steering ``layer`` by ``factor * steering_vector``.

    Returns the full output ids (prompt + continuation).
    """
    # NDIF returns nothing for steered runs that stop early; pin min_new_tokens
    # to max_new_tokens so the full continuation comes back.
    if "max_new_tokens" in generation_kwargs:
        generation_kwargs.setdefault(
            "min_new_tokens", generation_kwargs["max_new_tokens"]
        )

    out = None
    with model.generate(
        prompt, remote=remote, backend=backend, **generation_kwargs
    ) as tracer:
        if factor and layer is not None:
            with tracer.all():
                model.steer(
                    layers=layer,
                    steering_vector=steering_vector,
                    factor=float(factor),
                )
        out = model.generator.output.save()
    if out is None:
        raise RuntimeError(
            f"steered generation returned no output (layer={layer}, factor={factor}): "
            "the NDIF job completed but model.generator.output was empty."
        )
    return out


def generate_steered(
    model,
    prompt: str,
    layer: int,
    steering_vector: torch.Tensor,
    factors: Sequence[float],
    *,
    system: str | None = None,
    max_new_tokens: int = 120,
    remote: bool = True,
) -> dict[float, str]:
    """Generate ``prompt`` once per ``factor`` with the steering vector added.

    ``factor`` multiplies ``steering_vector`` at every generated position
    (``tracer.all()``); ``factor=0`` is the unsteered baseline. ``system``, if
    given, is placed in the system turn — e.g. an in-context persona to steer
    against. The chat template is always applied (via ``format_messages``, which
    also normalizes the system turn for models like gemma that lack one).

    Every run emits exactly ``max_new_tokens`` tokens (``min_new_tokens`` is
    pinned to ``max_new_tokens``), so the model can't stop early and
    steered/baseline continuations stay the same length and come back populated
    from NDIF.

    Returns ``{factor: continuation}``. The prompt is sliced off by token count
    (see :func:`_decode_continuation`).
    """
    text, prompt_token_count = _format_chat(model, prompt, system)

    outputs: dict[float, str] = {}
    for factor in factors:
        out = generate_steered_once(
            model,
            text,
            layer=layer,
            steering_vector=steering_vector,
            factor=factor,
            remote=remote,
            max_new_tokens=max_new_tokens,
        )
        outputs[float(factor)] = _decode_continuation(model, out, prompt_token_count)
    return outputs


# ── Band steering (multi-layer) + optional adaptive intensity ─────────────────
#
# The best-performing steering here: add each trait's own direction across a band
# of layers at a modest per-layer strength (compounds while staying
# in-distribution), optionally modulating the intensity over generation steps.
# The per-step schedules (linear taper / start-only) are the remote-feasible,
# dial-preserving slice of Scalena, Sarti & Nissim, *Multi-property Steering with
# Dynamic Activation Composition* (BlackboxNLP 2024,
# https://aclanthology.org/2024.blackboxnlp-1.34/); their full KL-adaptive DAC
# (Eq. 6) is a per-token feedback loop (local-only) and unnecessary here.


def band_steering_vectors(
    bands: dict | Sequence[dict], strength: float
) -> dict[int, torch.Tensor]:
    """Build ``{layer: combined_vector}`` for band steering at gap-unit ``strength``.

    ``bands`` is one ``{layer: info_dict}`` band (e.g. from
    :func:`persona_vectors.traits.load_trait_band`) or a list of them for
    multi-trait composition. Each layer's vector is
    ``Σ_trait steering_coefficient(info, strength) · unit_direction`` — the *base*
    magnitude the user dials in (``strength`` in gap units; ``strength=1`` = that
    layer's opposite-class centroid). Summing several bands per layer composes
    multiple traits (correlated traits reinforce). Feed the result to
    :func:`generate_band_steered`.
    """
    if isinstance(bands, dict):
        bands = [bands]
    out: dict[int, torch.Tensor] = {}
    for band in bands:
        for layer, info in band.items():
            term = steering_coefficient(info, strength) * info["unit_direction"]
            key = int(layer)
            out[key] = term if key not in out else out[key] + term
    return out


def dim_schedule(step: int, n_steps: int) -> float:
    """Per-step multiplier tapering linearly 1→0 (Scalena–Sarti–Nissim, Eq. 5)."""
    return max(0.0, 1.0 - step / max(n_steps - 1, 1))


def start_schedule(step: int, n_steps: int, frac: float = 1 / 3) -> float:
    """Per-step multiplier: full intensity for the first ``frac`` of steps, then 0.

    Pass bare as ``schedule=start_schedule``; for a different ``frac`` bind it with
    ``functools.partial(start_schedule, frac=0.5)``.
    """
    return 1.0 if step < max(1, int(n_steps * frac)) else 0.0


def generate_band_steered_once(
    model,
    prompt: str,
    layer_vectors: dict[int, torch.Tensor],
    *,
    schedule=None,
    remote: bool = True,
    max_new_tokens: int = 120,
) -> torch.Tensor:
    """One band-steered pass over an already-formatted ``prompt`` → full output ids.

    The band analogue of :func:`generate_steered_once` (the caller owns prompt
    formatting and decoding). Every layer in ``layer_vectors`` — each vector already
    carrying the dialled strength — is steered at every generated position, or
    per-step when ``schedule`` is given. ``model.steer`` is applied once per layer in
    ascending order (several steers on one layer raise nnsight's ``OutOfOrderError``).
    """
    layers = sorted(layer_vectors)
    with model.generate(
        prompt, remote=remote, max_new_tokens=max_new_tokens, min_new_tokens=max_new_tokens
    ) as tracer:
        if schedule is None:
            with tracer.all():
                for layer in layers:
                    model.steer(
                        layers=layer,
                        steering_vector=layer_vectors[layer],
                        factor=1.0,
                    )
        else:
            for step in range(max_new_tokens):
                factor = schedule(step, max_new_tokens)
                if factor:
                    with tracer.iter[step]:
                        for layer in layers:
                            model.steer(
                                layers=layer,
                                steering_vector=layer_vectors[layer],
                                factor=float(factor),
                            )
        out = model.generator.output.save()
    return out


def generate_band_steered(
    model,
    prompt: str,
    layer_vectors: dict[int, torch.Tensor],
    *,
    schedule=None,
    system: str | None = None,
    max_new_tokens: int = 120,
    remote: bool = True,
) -> str:
    """Current best steering: a band of layers, optionally with adaptive intensity.

    ``layer_vectors`` is ``{layer: vector}`` (e.g. from
    :func:`band_steering_vectors`) — each vector already carries the user-dialled
    *base* strength. Every layer is steered at every generated position; using each
    layer's own direction at a modest per-layer strength compounds across the band
    while staying in-distribution — far stronger and more coherent than a large
    single-layer push.

    ``schedule`` shapes the intensity **over generation steps** as a multiplier on
    that base — a ``schedule(step, n_steps) -> [0, 1]`` callable, called internally
    with ``n_steps=max_new_tokens`` so it always matches the step count: ``None``
    keeps it constant (multi-layer, fixed — a drop-in for the single-layer fixed
    steering, just stronger); :func:`dim_schedule` tapers it 1→0 to protect fluency
    when steering hard; :func:`start_schedule` steers only the opening tokens. The
    user keeps the strength dial *and* gains an adaptive multiplier. Returns the
    continuation string.
    """
    text, prompt_token_count = _format_chat(model, prompt, system)
    out = generate_band_steered_once(
        model,
        text,
        layer_vectors,
        schedule=schedule,
        remote=remote,
        max_new_tokens=max_new_tokens,
    )
    return _decode_continuation(model, out, prompt_token_count)
