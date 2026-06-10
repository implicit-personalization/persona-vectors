"""Attribute co-occurrence over the persona population.

How often do two persona attributes move together in the sampled dataset? This
is the *confound map* a difference-of-means trait direction has to be read
against: if ``religion`` and ``religion_at_16`` co-occur strongly, a steering
axis built for one will absorb the other.

We use **Cramér's V**, the standard association measure for nominal variables of
any cardinality (symmetric, in ``[0, 1]``: 0 = independent, 1 = one determines the
other), with the **Bergsma (2013) bias correction** applied via SciPy (the plain
estimator is positively biased). Numeric attributes (e.g. ``age``) are
quantile-binned first so the same measure applies to everything.

  - Bergsma (2013): https://doi.org/10.1016/j.jkss.2012.10.002
  - SciPy: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.contingency.association.html

The attribute-less ``baseline_assistant`` persona is excluded by default: it has no
attributes, and including its all-missing row creates an aligned singleton that
blows up chi-square and inflates every pair (e.g. ``sex``↔``age`` 0.71 vs the true
~0.05). Real personas are otherwise complete.
"""

from collections.abc import Sequence
from itertools import combinations
from typing import Any

import numpy as np
from persona_data.synth_persona import BASELINE_PERSONA_ID
from scipy.stats.contingency import association, crosstab


def _discretize(values: Sequence[Any], kind: str, n_bins: int = 4) -> np.ndarray:
    """Discrete string labels for one attribute column.

    Categorical / binary / ordinal values are already discrete. ``numeric`` values
    (e.g. ``age``) are quantile-binned into ``n_bins`` buckets so the same measure
    applies to everything. ``unique`` on the quantile edges collapses ties, so
    low-variance fields just yield fewer bins instead of empty ones.
    """
    if kind != "numeric":
        return np.asarray([str(v) for v in values], dtype=object)
    nums = np.asarray(values, dtype=float)
    edges = np.unique(np.quantile(nums, np.linspace(0, 1, n_bins + 1)))
    return np.digitize(nums, edges[1:-1]).astype(str)


def _cramers_v(x: np.ndarray, y: np.ndarray) -> float:
    """Bias-corrected Cramér's V (Bergsma 2013) between two label arrays, in ``[0, 1]``.

    Returns ``nan`` when either variable is constant (``min(r, k) < 2``), where V is
    undefined. See module docstring for references.
    """
    table = crosstab(x, y).count
    if min(table.shape) < 2:
        return float("nan")
    return float(association(table, method="cramer", correction=True))


def attribute_association_matrix(
    dataset: Any,
    attributes: Sequence[str] | None = None,
    persona_ids: Sequence[str] | None = None,
    *,
    n_bins: int = 4,
) -> tuple[list[str], np.ndarray]:
    """Pairwise Cramér's V co-occurrence matrix over persona attributes.

    ``attributes`` defaults to ``dataset.attribute_names`` (already excludes
    identifier / dropped fields). High-cardinality nominals (``city``, ``state``)
    give unstable V, so pass an explicit list to include them deliberately.

    ``persona_ids`` defaults to every persona **except the attribute-less
    ``baseline_assistant``** — it has no attributes, so including it would only add
    ``<missing>`` noise to a population analysis. Pass an explicit list to override.

    Returns ``(labels, matrix)`` where ``labels`` are the (short) attribute names
    and ``matrix`` is a symmetric ``(A, A)`` ``float`` array with diagonal ``1.0``
    (``nan`` for pairs involving a constant attribute).
    """
    attrs = (
        list(attributes) if attributes is not None else list(dataset.attribute_names)
    )
    if persona_ids is None:
        persona_ids = [pid for pid in dataset.persona_ids if pid != BASELINE_PERSONA_ID]

    columns = [
        _discretize(
            dataset.attribute_values(attr, persona_ids),
            dataset.attribute_info(attr).get("kind", "categorical"),
            n_bins=n_bins,
        )
        for attr in attrs
    ]

    matrix = np.eye(len(attrs), dtype=float)
    for i, j in combinations(range(len(attrs)), 2):
        matrix[i, j] = matrix[j, i] = _cramers_v(columns[i], columns[j])

    return attrs, matrix


def top_cooccurring_pairs(
    labels: Sequence[str], matrix: np.ndarray, k: int = 10
) -> list[tuple[str, str, float]]:
    """Return the ``k`` highest off-diagonal attribute pairs as ``(a, b, v)``."""
    pairs = [
        (labels[i], labels[j], float(matrix[i, j]))
        for i, j in combinations(range(len(labels)), 2)
        if np.isfinite(matrix[i, j])
    ]
    pairs.sort(key=lambda p: p[2], reverse=True)
    return pairs[:k]
