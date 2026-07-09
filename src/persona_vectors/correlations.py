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
from scipy.stats import rankdata, spearmanr
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


def off_diagonal_pair_values(
    left: np.ndarray, right: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Aligned finite off-diagonal values from two square matrices."""
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.shape != right.shape or left.ndim != 2 or left.shape[0] != left.shape[1]:
        raise ValueError(
            f"expected two square matrices with the same shape, got {left.shape} and {right.shape}"
        )
    idx = np.triu_indices(left.shape[0], k=1)
    x = left[idx]
    y = right[idx]
    finite = np.isfinite(x) & np.isfinite(y)
    return x[finite], y[finite]


def rank_delta_matrix(
    trait_similarity: np.ndarray, cooccurrence: np.ndarray
) -> np.ndarray:
    """Off-diagonal rank-percentile difference between two association matrices.

    Raw ``|cosine| - Cramér's V`` is visually useful but not metric-faithful:
    cosine similarity and Cramér's V do not share units. This converts each
    matrix's finite off-diagonal entries to within-matrix percentile ranks and
    returns ``rank(|cosine|) - rank(V)`` in ``[-1, 1]``. Positive cells are more
    prominent in representation geometry than in dataset co-occurrence; negative
    cells are stronger co-occurrences than geometry would suggest.
    """
    trait_similarity = np.asarray(trait_similarity, dtype=float)
    cooccurrence = np.asarray(cooccurrence, dtype=float)
    if (
        trait_similarity.shape != cooccurrence.shape
        or trait_similarity.ndim != 2
        or trait_similarity.shape[0] != trait_similarity.shape[1]
    ):
        raise ValueError(
            "trait_similarity and cooccurrence must be square matrices with the same shape"
        )

    n = trait_similarity.shape[0]
    idx = np.triu_indices(n, k=1)
    trait_vals = trait_similarity[idx]
    co_vals = cooccurrence[idx]
    finite = np.isfinite(trait_vals) & np.isfinite(co_vals)
    if finite.sum() < 2:
        return np.full_like(trait_similarity, np.nan, dtype=float)

    denom = float(finite.sum() - 1)
    trait_pct = (rankdata(trait_vals[finite], method="average") - 1.0) / denom
    co_pct = (rankdata(co_vals[finite], method="average") - 1.0) / denom

    out = np.full((n, n), np.nan, dtype=float)
    out[np.diag_indices(n)] = 0.0
    rows = idx[0][finite]
    cols = idx[1][finite]
    values = trait_pct - co_pct
    out[rows, cols] = values
    out[cols, rows] = values
    return out


def matrix_spearman(
    trait_similarity: np.ndarray, cooccurrence: np.ndarray
) -> tuple[float, float, int]:
    """Spearman correlation over aligned finite off-diagonal matrix entries."""
    x, y = off_diagonal_pair_values(trait_similarity, cooccurrence)
    rho, p = spearmanr(x, y)
    return float(rho), float(p), int(len(x))


def matrix_permutation_test(
    trait_similarity: np.ndarray,
    cooccurrence: np.ndarray,
    *,
    n_perm: int = 4999,
    seed: int = 0,
) -> tuple[float, float]:
    """Mantel/QAP-style permutation test for two attribute-pair matrices.

    The off-diagonal matrix entries are not independent because every attribute
    appears in many pairs. This test keeps one matrix fixed, repeatedly permutes
    the attribute labels of the other matrix (same permutation for rows and
    columns), and recomputes Spearman correlation. The returned two-sided
    empirical p-value asks how often a random relabeling yields an association at
    least as large as the observed one.
    """
    trait_similarity = np.asarray(trait_similarity, dtype=float)
    cooccurrence = np.asarray(cooccurrence, dtype=float)
    observed, _, _ = matrix_spearman(trait_similarity, cooccurrence)
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(cooccurrence.shape[0])
        shuffled = cooccurrence[perm][:, perm]
        rho, _, _ = matrix_spearman(trait_similarity, shuffled)
        if abs(rho) >= abs(observed):
            count += 1
    p = (count + 1.0) / (n_perm + 1.0)
    return float(observed), float(p)
