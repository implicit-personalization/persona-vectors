"""Deconfounded trait vectors from minimal-pair attribute swaps.

A persona vector mixes every attribute a persona happens to carry, so a
population difference-of-means direction for one attribute absorbs whatever
co-occurs with it. A *trait* vector isolates a single attribute by swapping only
that attribute on a persona (``persona_data.templated.swap_attribute``,
re-rendering the whole view so coupled sentences stay coherent), extracting both
views, and taking the within-pair activation delta. Averaging those paired
deltas over personas cancels everything that did not change.

This builds the **description-level** flavor (``PERSONA_MEAN`` over the swapped
``templated_view``); the answer-level flavor (force-decoded explicit answer,
``ANSWER_MEAN``) reuses the same orientation logic with a different mask.
:func:`build_trait_direction` returns a steering-harness direction dict
(``layer``, ``unit_direction``, ``gap_norm``, ``auc``, ``positive``, …), so trait
vectors plug straight into :func:`persona_vectors.steering.generate_steered` /
:func:`persona_vectors.steering.steering_coefficient`.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from persona_data.prompts import format_prompt
from persona_data.synth_persona import PersonaData, QAPair
from persona_data.templated import swap_attribute
from sklearn.metrics import roc_auc_score

from persona_vectors.activations import extract_activations
from persona_vectors.attributes import attribute_schema
from persona_vectors.extraction import MaskStrategy, prepare_inputs_for_strategy


@dataclass
class TraitDeltas:
    """Paired activations for one swapped attribute, oriented by *value*.

    ``acts_from`` / ``acts_to`` are ``(n_personas, num_layers, hidden)`` and are
    keyed by attribute value (the ``value_from`` rep vs the ``value_to`` rep),
    not by which value was original — so per-persona deltas point the same way
    and averaging is meaningful.
    """

    attribute: str
    variant: str
    value_from: str
    value_to: str
    persona_ids: list[str]
    acts_from: np.ndarray
    acts_to: np.ndarray

    @property
    def deltas(self) -> np.ndarray:
        """Per-persona ``act(value_to) - act(value_from)``."""
        return self.acts_to - self.acts_from

    @property
    def mean_delta(self) -> np.ndarray:
        """Mean swap delta per layer, ``(num_layers, hidden)`` (the trait vector)."""
        return self.deltas.mean(0).astype(np.float32)

    def layer_stats(self) -> dict[int, dict[str, float]]:
        """Per-layer ``{auc, act_norm}`` for the paired ``from``/``to`` activations.

        ``auc`` is how well projecting both values onto the mean unit delta
        separates them; ``act_norm`` is the typical residual-stream norm.
        """
        n = len(self.persona_ids)
        y = np.concatenate([np.zeros(n), np.ones(n)])  # 0 = value_from, 1 = value_to
        stats: dict[int, dict[str, float]] = {}
        for layer in range(self.acts_from.shape[1]):
            x_from = self.acts_from[:, layer, :]
            x_to = self.acts_to[:, layer, :]
            stacked = np.concatenate([x_from, x_to])
            d = (x_to - x_from).mean(0)
            unit = d / (np.linalg.norm(d) + 1e-12)
            stats[layer] = {
                "auc": float(roc_auc_score(y, stacked @ unit)),
                "act_norm": float(np.linalg.norm(stacked, axis=1).mean()),
            }
        return stats


def binary_attribute_values(persona_dataset, attribute: str) -> tuple[str, str]:
    """Return the two values of a binary attribute as ``(value_from, value_to)``."""
    info = attribute_schema(persona_dataset).get(attribute, {})
    known = info.get("ordered_values") or [
        v["value"] for v in info.get("seed_values_sorted_by_count", [])
    ]
    if info.get("kind") != "binary" or len(known) != 2:
        raise ValueError(f"{attribute!r} is not a 2-value binary attribute: {known}")
    return str(known[0]), str(known[1])


def attribute_contrast_values(persona_dataset, attribute: str) -> tuple[object, object]:
    """Return the ``(value_from, value_to)`` pole pair to contrast for ``attribute``.

    The minimal-pair extraction needs two values to swap between. How those are
    chosen depends on the attribute *kind* (only ordered kinds have a meaningful
    single direction, so unordered ``nominal`` attributes are rejected):

    * ``binary`` — the two values (same as :func:`binary_attribute_values`).
    * ``ordinal`` — the two **extremes** of ``ordered_values`` (e.g. *Less than
      $5,000* → *Above $10 million*, *Extremely liberal* → *Extremely
      conservative*), so ``+`` points from the bottom rung to the top.
    * ``numeric`` — the min/max of the seed values (e.g. ``age`` 18 → 89).

    ``value_to`` is the high/``+`` pole; the trait direction points ``from`` → ``to``.
    """
    info = attribute_schema(persona_dataset).get(attribute, {})
    kind = info.get("kind")

    if kind == "binary":
        return binary_attribute_values(persona_dataset, attribute)

    if kind == "ordinal":
        ordered = info.get("ordered_values") or []
        if len(ordered) < 2:
            raise ValueError(f"{attribute!r} ordinal has <2 ordered_values: {ordered}")
        return str(ordered[0]), str(ordered[-1])

    if kind == "numeric":
        seeds = [v["value"] for v in info.get("seed_values_sorted_by_count", [])]
        nums = sorted(float(v) for v in seeds)
        if len(nums) < 2:
            raise ValueError(f"{attribute!r} numeric has <2 seed values")
        # Keep the original dtype where possible (ints render cleaner than 18.0).
        cast = int if all(float(n).is_integer() for n in (nums[0], nums[-1])) else float
        return cast(nums[0]), cast(nums[-1])

    raise ValueError(
        f"{attribute!r} kind {kind!r} has no single contrast direction "
        "(only binary/ordinal/numeric are supported)"
    )


def _render_at_pole(
    persona_dataset, persona: PersonaData, attribute: str, value: object
):
    """Return ``persona`` re-rendered with ``attribute == value`` (minimal pair pole).

    Reuses :func:`swap_attribute` (which validates the v4.0 template and only
    re-renders sentences tied to ``attribute``). If the persona already sits at
    ``value`` there is nothing to swap, so the original is returned unchanged.
    """
    if str(persona.persona.get(attribute)) == str(value):
        return persona
    _, swapped = swap_attribute(persona_dataset, persona.id, attribute, new_value=value)
    return swapped


def _persona_vectors(
    model,
    persona: PersonaData,
    qa_pairs: list[QAPair],
    *,
    variant: str,
    mask_strategy: MaskStrategy,
    remote: bool,
    on_status: Callable | None,
    backend_factory: Callable[[], object] | None,
) -> tuple[np.ndarray, list]:
    """The inner half of ``run_extraction`` for one persona: vectors, no save.

    Returns the ``(num_layers, hidden)`` masked-mean activation and the prepared
    inputs (so callers can preview the averaged tokens).
    """
    prepared = prepare_inputs_for_strategy(
        tokenizer=model.tokenizer,
        system_prompt=format_prompt(persona, variant),
        qa_pairs=qa_pairs,
        mask_strategy=mask_strategy,
    )
    vectors = extract_activations(
        model,
        input_ids_list=[p.input_ids for p in prepared],
        token_masks=[p.token_mask for p in prepared],
        remote=remote,
        on_status=on_status,
        backend_factory=backend_factory,
    )
    return vectors.float().cpu().numpy(), prepared


def extract_trait_deltas(
    model,
    persona_dataset,
    attribute: str,
    runs: Sequence[tuple[PersonaData, list[QAPair]]],
    *,
    variant: str = "templated",
    mask_strategy: MaskStrategy = MaskStrategy.PERSONA_MEAN,
    remote: bool = False,
    on_status: Callable | None = None,
    backend_factory: Callable[[], object] | None = None,
    verbose: bool = False,
) -> TraitDeltas:
    """Extract value-oriented minimal-pair activations for one attribute.

    Picks a contrast pole pair with :func:`attribute_contrast_values` (binary →
    the two values; ordinal/numeric → the two extremes, e.g. poor↔rich,
    young↔old) and, **for every persona**, re-renders the templated view at both
    poles and extracts both. Rendering both poles explicitly (rather than
    swapping off whatever the persona happened to be) means a mid-scale ordinal
    persona still contributes the full extreme→extreme delta, and every
    per-persona delta is oriented the same way (``+`` → ``value_to``) so
    averaging is meaningful. ``qa_pairs`` only builds the prompt — ``PERSONA_MEAN``
    averages the persona prefix, so the question content is irrelevant. Personas
    whose ``templated_view`` is not the swappable v4.0 render (e.g. the
    attribute-less ``baseline_assistant``) are skipped.
    """
    value_from, value_to = attribute_contrast_values(persona_dataset, attribute)

    ids: list[str] = []
    from_rows: list[np.ndarray] = []
    to_rows: list[np.ndarray] = []
    for persona, qa_pairs in runs:
        if not qa_pairs:
            continue
        try:
            view_from = _render_at_pole(persona_dataset, persona, attribute, value_from)
            view_to = _render_at_pole(persona_dataset, persona, attribute, value_to)
        except (KeyError, ValueError) as exc:
            if verbose:
                print(f"  skip {persona.id}: {exc}")
            continue

        if verbose:
            from persona_vectors.preview import preview_trait_contrast

            preview_trait_contrast(
                attribute,
                str(value_from),
                str(value_to),
                view_from.templated_view,
                view_to.templated_view,
            )

        kwargs = dict(
            variant=variant,
            mask_strategy=mask_strategy,
            remote=remote,
            on_status=on_status,
            backend_factory=backend_factory,
        )
        from_vec, from_prepared = _persona_vectors(model, view_from, qa_pairs, **kwargs)
        to_vec, _ = _persona_vectors(model, view_to, qa_pairs, **kwargs)

        # Show the averaged region once, on the first kept persona.
        if verbose and not ids:
            from persona_vectors.preview import preview_prepared_inputs

            preview_prepared_inputs(
                from_prepared,
                tokenizer=model.tokenizer,
                variant=variant,
                mask_strategy=mask_strategy,
            )

        from_rows.append(from_vec)
        to_rows.append(to_vec)
        ids.append(persona.id)

    if not ids:
        raise ValueError(f"no swappable personas for attribute {attribute!r}")

    return TraitDeltas(
        attribute=attribute,
        variant=variant,
        value_from=str(value_from),
        value_to=str(value_to),
        persona_ids=ids,
        acts_from=np.stack(from_rows),
        acts_to=np.stack(to_rows),
    )


def _trait_direction_dict(
    mean_delta_row: np.ndarray,
    *,
    attribute: str,
    variant: str,
    positive: str,
    layer: int,
    auc: float,
    act_norm: float,
    n_personas: int,
) -> dict:
    """Steering-ready direction dict from one layer's mean delta.

    Schema consumed by :func:`persona_vectors.steering.generate_steered` /
    :func:`persona_vectors.steering.steering_coefficient` (``layer``,
    ``unit_direction``, ``gap_norm``, ``auc``, ``positive``, …).
    """
    d = np.asarray(mean_delta_row, dtype=np.float32)
    unit = d / (np.linalg.norm(d) + 1e-12)
    return {
        "attribute": attribute,
        "variant": variant,
        "task": "binary",
        "positive": positive,
        "layer": int(layer),
        "auc": float(auc),
        "n_personas": int(n_personas),
        "gap_norm": float(np.linalg.norm(d)),
        "act_norm": float(act_norm),
        "direction": torch.from_numpy(d),
        "unit_direction": torch.from_numpy(unit.astype(np.float32)),
    }


def build_trait_direction(
    deltas: TraitDeltas, *, candidate_layers: Sequence[int]
) -> dict:
    """Mean minimal-pair delta at the best-separating layer.

    The direction is the mean of ``act(value_to) - act(value_from)`` over
    personas (so ``+`` points to ``value_to``); the chosen layer maximises how
    well projecting the paired ``from``/``to`` activations onto that direction
    separates the two values (``|AUC - 0.5|``, the same criterion as the
    population difference-of-means builder, but on minimal pairs). Returns a
    steering-harness direction dict so the result drops into the steering flow.
    """
    stats = deltas.layer_stats()
    mean = deltas.mean_delta
    best = max(candidate_layers, key=lambda layer: abs(stats[layer]["auc"] - 0.5))
    return _trait_direction_dict(
        mean[best],
        attribute=deltas.attribute,
        variant=deltas.variant,
        positive=deltas.value_to,
        layer=best,
        auc=stats[best]["auc"],
        act_norm=stats[best]["act_norm"],
        n_personas=len(deltas.persona_ids),
    )
