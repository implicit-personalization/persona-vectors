"""Causal steering on NDIF-hosted models from precomputed persona vectors.

Build a difference-of-means steering vector from precomputed ``templated``
activations, then add it during generation on the live model (remote NDIF) and
read off the behavioral shift.

Memory note is feasibility: gemma-3-27b can't run here, but Llama-3.1-405B is
hosted on NDIF, so we steer *that* model using its own precomputed 405B vectors.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from persona_data.prompts import format_messages
from sklearn.decomposition import PCA, FastICA
from sklearn.metrics import roc_auc_score

from persona_vectors.artifacts import PersonaVectorSource
from persona_vectors.probes import attribute_probe_labels

# How generation-steering works (nnterp / nnsight):
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
# additive hook on the layer's residual output: h <- h + factor * vector. The
# direction ``v`` is the difference-of-means persona axis; ``factor`` is the
# push strength. Steering at every step is what makes the shift persist through
# the whole continuation rather than decaying after the first token.


@dataclass
class SteeringSpec:
    """One additive steering intervention: ``h += coefficient * vector`` at ``layer``.

    ``vector`` is a 1-D residual-space direction (hidden_size); pass a *unit*
    direction and set ``coefficient`` to the desired push strength.
    """

    layer: int
    vector: torch.Tensor
    coefficient: float


def steering_coefficient(info: dict, strength: float, *, sign: float = 1.0) -> float:
    """Calibrated steering coefficient in *gap units* from a direction ``info``.

    ``info`` is a :func:`build_attribute_direction` / :func:`build_ica_direction`
    result. ``strength`` is in gap units: ``strength=1`` lands the activation at
    the opposite-class centroid (``coefficient = strength * gap_norm``). Scaling by
    the residual *norm* (``act_norm``) instead over-steers 8-28x the real class
    separation and collapses the model off-manifold. ``sign`` flips the causal
    polarity: a decode-optimal difference-of-means axis can steer the *opposite*
    way behaviourally, so pass ``sign=-1`` to invert (see the per-attribute
    calibration in the steering notebooks / persona-ui).
    """
    return float(strength * sign * info["gap_norm"])


def build_steering_spec(
    info: dict, strength: float, *, sign: float = 1.0
) -> SteeringSpec:
    """A ready, gap-unit-calibrated :class:`SteeringSpec` from a direction ``info``.

    Single source of truth shared by the steering notebooks and persona-ui so the
    coefficient formula can't drift. Steers ``info``'s unit direction at its layer;
    to steer a different layer, rebuild ``info`` at that layer (the direction is
    layer-specific). See :func:`steering_coefficient` for the ``strength``/``sign``
    semantics.
    """
    return SteeringSpec(
        layer=int(info["layer"]),
        vector=info["unit_direction"],
        coefficient=steering_coefficient(info, strength, sign=sign),
    )


def apply_steering(model, tracer, spec: "SteeringSpec | None") -> None:
    """Register a steering spec on an open ``model.generate`` tracer.

    Call this *inside* a ``with model.generate(...) as tracer:`` block, before
    saving the output. ``None`` or zero-coefficient specs are skipped, so callers
    can wire it in unconditionally. ``tracer.all()`` scopes the additive hook to
    every generated position (not just the prompt's last token).
    """
    if spec is None or not spec.coefficient:
        return
    with tracer.all():
        model.steer(
            layers=spec.layer,
            steering_vector=spec.vector,
            factor=float(spec.coefficient),
        )


def _class_name(names, idx: int) -> str:
    return str(names[idx]) if names and idx < len(names) else f"class {idx}"


def _contrast_labels(
    attr,
    target_class: int | None = None,
    negative_class: int | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Map any attribute to a 0/1 contrast for a difference-of-means direction.

    The method depends on the attribute type, so every attribute gets a single
    signed steering axis (not just binary ones):

    - ``binary``: used as-is; ``+`` = class 1.
    - ``numeric`` / ``ordinal``: median split; ``+`` = above the median (e.g.
      older, more educated, wealthier).
    - ``categorical`` (nominal, k>2): one-vs-rest; ``+`` = ``target_class``
      (default: the most frequent class). Pass ``negative_class`` as well for a
      **class-vs-class** contrast (e.g. Protestant vs Catholic): only personas in
      those two classes are kept and the axis points target → away-from-negative.
      High-cardinality nominals (city, state) give weak directions because each
      class has few personas — caveat, not a hard error.

    Returns ``(y01, keep_mask, positive_label)``. ``keep_mask`` selects which
    personas enter the difference of means (all of them except for class-vs-class
    categorical contrasts, which drop the other classes).
    """
    y = np.asarray(attr.y)
    names = getattr(attr, "class_names", None)
    keep = np.ones(y.shape[0], dtype=bool)
    if attr.task == "binary":
        pos = names[1] if names and len(names) > 1 else "class 1"
        return y.astype(int), keep, str(pos)
    if attr.task in {"numeric", "ordinal"}:
        med = np.median(y)
        z = (y > med).astype(int)
        if z.min() == z.max():  # median at an extreme
            z = (y >= med).astype(int)
        return z, keep, "above median"
    # categorical
    if target_class is None:
        vals, counts = np.unique(y, return_counts=True)
        target_class = int(vals[counts.argmax()])
    if negative_class is not None:
        keep = (y == target_class) | (y == negative_class)
        label = f"{_class_name(names, target_class)} vs {_class_name(names, negative_class)}"
    else:
        label = _class_name(names, target_class)  # one-vs-rest
    return (y == target_class).astype(int), keep, str(label)


def build_attribute_direction(
    store: PersonaVectorSource,
    persona_dataset,
    attribute: str,
    *,
    variant: str = "templated",
    candidate_layers: Sequence[int],
    persona_ids: Sequence[str] | None = None,
    mask_strategy: object | None = None,
    target_class: int | None = None,
    negative_class: int | None = None,
) -> dict:
    """Difference-of-means steering direction at the best-separating layer.

    Loads each persona's activation once, slices only ``candidate_layers`` (so
    memory stays tiny vs. the full layer stack), maps the attribute to a 0/1
    contrast (median split for numeric/ordinal, one-vs-rest for categorical; see
    :func:`_contrast_labels`), and picks the layer whose difference-of-means
    projection best separates the two halves (|AUC - 0.5|). Returns the raw
    direction (so a steering ``factor`` is in units of mean gaps), the unit
    direction, the chosen layer, the typical residual-stream norm, and
    ``positive`` (what the ``+`` direction means).
    """
    ids = list(persona_ids or store.list_personas([variant], include_baseline=False))
    layers = list(candidate_layers)

    rows: list[np.ndarray] = []
    for pid in ids:
        acts = (
            store.load(variant, pid, mask_strategy=mask_strategy).float().cpu().numpy()
        )
        rows.append(acts[layers])  # (n_candidate_layers, hidden)
    sliced = np.stack(rows, axis=0)  # (n_personas, n_layers, hidden)

    attr = attribute_probe_labels(persona_dataset, attribute, ids)
    y01, keep, positive = _contrast_labels(attr, target_class, negative_class)
    if keep.sum() < 4 or y01[keep].min() == y01[keep].max():
        raise ValueError(
            f"attribute {attribute!r}: contrast has too few / single class"
        )
    sliced = sliced[keep]
    y01 = y01[keep]

    best = None
    for li, layer in enumerate(layers):
        Xl = sliced[:, li, :]
        d = Xl[y01 == 1].mean(0) - Xl[y01 == 0].mean(0)
        proj = Xl @ d
        auc = roc_auc_score(y01, proj)
        sep = abs(auc - 0.5)
        if best is None or sep > best["sep"]:
            best = {
                "layer": int(layer),
                "sep": float(sep),
                "auc": float(auc),
                "direction": d.astype(np.float32),
                "act_rms": float(np.sqrt((Xl**2).mean())),
                "act_norm": float(np.linalg.norm(Xl, axis=1).mean()),
            }

    d = best["direction"]
    unit = d / (np.linalg.norm(d) + 1e-12)
    return {
        "attribute": attribute,
        "variant": variant,
        "task": attr.task,
        "positive": positive,
        "layer": best["layer"],
        "auc": best["auc"],
        "n_personas": int(y01.shape[0]),
        "gap_norm": float(np.linalg.norm(d)),
        "act_norm": best["act_norm"],
        "direction": torch.from_numpy(d),
        "unit_direction": torch.from_numpy(unit.astype(np.float32)),
    }


def build_ica_direction(
    store: PersonaVectorSource,
    persona_dataset,
    attribute: str,
    *,
    variant: str = "templated",
    candidate_layers: Sequence[int],
    persona_ids: Sequence[str] | None = None,
    mask_strategy: object | None = None,
    target_class: int | None = None,
    negative_class: int | None = None,
    n_pca: int = 50,
    n_ica: int = 30,
    seed: int = 0,
) -> dict:
    """Unsupervised (ICA) steering direction at the best-separating layer.

    Same contract and return shape as :func:`build_attribute_direction`, but the
    direction is an *independent component* (FastICA on PCA-whitened activations),
    discovered without labels. The attribute labels are used only to pick which
    (layer, component) best separates the contrast, to orient the sign, and to set
    the mean-gap scale - so the returned ``gap_norm`` / ``direction`` are in the
    same units as the difference-of-means direction and steering factors are
    directly comparable. Useful to ask: does an unsupervised direction steer like
    the supervised one?
    """
    ids = list(persona_ids or store.list_personas([variant], include_baseline=False))
    layers = list(candidate_layers)

    rows: list[np.ndarray] = []
    for pid in ids:
        acts = (
            store.load(variant, pid, mask_strategy=mask_strategy).float().cpu().numpy()
        )
        rows.append(acts[layers])
    sliced = np.stack(rows, axis=0)  # (n_personas, n_layers, hidden)

    attr = attribute_probe_labels(persona_dataset, attribute, ids)
    y01, keep, positive = _contrast_labels(attr, target_class, negative_class)
    if keep.sum() < 4 or y01[keep].min() == y01[keep].max():
        raise ValueError(
            f"attribute {attribute!r}: contrast has too few / single class"
        )
    sliced = sliced[keep]
    y01 = y01[keep]

    best = None
    for li, layer in enumerate(layers):
        Xl = sliced[:, li, :]
        Xc = Xl - Xl.mean(0)
        # ICA is data-hungry: cap PCA-whitening dims well below the sample count
        # (~n/8) so FastICA stays stable for small persona sets (e.g. 100).
        n_comp = min(n_pca, Xc.shape[1], max(2, Xc.shape[0] // 8))
        pca = PCA(n_components=n_comp, random_state=seed).fit(Xc)
        scores = pca.transform(Xc)
        n_src = min(n_ica, n_comp)
        ica = FastICA(
            n_components=n_src,
            random_state=seed,
            max_iter=800,
            tol=1e-3,
            whiten="unit-variance",
        ).fit(scores)
        sources = ica.transform(scores)
        for c in range(n_src):
            sep = abs(roc_auc_score(y01, sources[:, c]) - 0.5)
            if best is None or sep > best["sep"]:
                # Map the component back to a raw residual-space direction.
                direction = pca.components_.T @ ica.components_[c]
                best = {"layer": int(layer), "sep": sep, "Xl": Xl, "u": direction}

    Xl = best["Xl"]
    unit = best["u"] / (np.linalg.norm(best["u"]) + 1e-12)
    if roc_auc_score(y01, Xl @ unit) < 0.5:  # orient: + points to class 1
        unit = -unit
    proj = Xl @ unit
    gap = float(proj[y01 == 1].mean() - proj[y01 == 0].mean())
    direction = (unit * gap).astype(np.float32)
    return {
        "attribute": attribute,
        "variant": variant,
        "task": attr.task,
        "positive": positive,
        "layer": best["layer"],
        "auc": float(roc_auc_score(y01, proj)),
        "n_personas": int(y01.shape[0]),
        "gap_norm": float(abs(gap)),
        "act_norm": float(np.linalg.norm(Xl, axis=1).mean()),
        "direction": torch.from_numpy(direction),
        "unit_direction": torch.from_numpy(unit.astype(np.float32)),
    }


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

    # NOTE: This workaround is required due to unexpected behavior in NDIF.
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

    Every run emits exactly ``max_new_tokens`` tokens (``min_new_tokens`` is pinned to ``max_new_tokens``),
    so the model can't stop early and steered/baseline continuations stay the same length and come back populated from NDIF.

    Returns ``{factor: continuation}``. The prompt is sliced off by token count
    (``response_start_idx`` from ``format_messages``) rather than by string
    matching: decoding with ``skip_special_tokens`` drops the chat-template
    markers, so a string prefix test fails and the prompt leaks into the output.
    """
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    text, prompt_token_count = format_messages(
        messages, model.tokenizer, add_generation_prompt=True
    )

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
        outputs[float(factor)] = model.tokenizer.decode(
            out[0][prompt_token_count:], skip_special_tokens=True
        ).strip()
    return outputs
