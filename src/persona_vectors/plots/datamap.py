"""Interactive datamapplot view of a 2-D persona embedding.

``attribute_cluster_labels`` names density clusters (e.g. HDBSCAN output) by
their over-represented attribute values; ``persona_datamap`` renders the
embedding as a zoomable deck.gl data map with per-persona attribute hover
text and search. Requires ``datamapplot`` (dev dependency group).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

# Values that are meaningless without their attribute name ("race: Other").
_AMBIGUOUS_VALUES = {"Other", "None", "Yes", "No"}


def attribute_cluster_labels(
    cluster_ids: np.ndarray,
    attribute_values: dict[str, Sequence[str]],
    *,
    top_k: int = 2,
    min_share: float = 0.5,
    min_z: float = 3.0,
    include_details: bool = False,
    noise_label: str = "Unlabelled",
):
    """Name clusters by their most over-represented attribute values.

    A value can appear in a cluster's name only if it is the cluster majority
    (in-cluster share >= ``min_share``) AND that share beats the value's
    dataset base rate by a one-proportion z-test (z >= ``min_z``). The base
    rate makes the bar chance-aware: a binary value needs a near-unanimous
    cluster, while a rare value (1-in-10 base rate) qualifies at much lower
    shares. Qualifying values are scored ``share * log(share / base_rate)``
    and the ``top_k`` strongest form the name (rendered in
    ``attribute_values`` key order). When two clusters would share a name,
    each appends its next-best value until the names differ. Cluster id
    ``-1`` is treated as noise.

    Names are summaries, not membership guarantees, so with
    ``include_details=True`` a second return value maps each name to a
    purity-vs-baseline line ("Female 100% (vs 54% overall) · Never married
    52% (vs 37% overall)") for use in hover text or figure captions.
    """
    cluster_ids = np.asarray(cluster_ids)
    values = {
        name: np.asarray(vals, dtype=object) for name, vals in attribute_values.items()
    }
    global_share = {
        name: {value: float((vals == value).mean()) for value in np.unique(vals)}
        for name, vals in values.items()
    }

    def ranked_parts(mask: np.ndarray) -> list[tuple[float, str, str, float, float]]:
        parts = []
        for name, vals in values.items():
            uniq, counts = np.unique(vals[mask], return_counts=True)
            top = uniq[counts.argmax()]
            share = counts.max() / mask.sum()
            base = global_share[name][top]
            if base >= 1.0:  # constant attribute, never informative
                continue
            z = (share - base) / np.sqrt(base * (1.0 - base) / mask.sum())
            if share >= min_share and z >= min_z:
                text = f"{name}: {top}" if top in _AMBIGUOUS_VALUES else str(top)
                parts.append((share * np.log(share / base), name, text, share, base))
        parts.sort(reverse=True)
        return parts

    attr_order = list(values)

    def ordered(parts):
        return sorted(parts, key=lambda part: attr_order.index(part[1]))

    masks = {cid: cluster_ids == cid for cid in np.unique(cluster_ids) if cid != -1}
    cluster_parts = {cid: ranked_parts(mask) for cid, mask in masks.items()}
    n_used = {cid: min(top_k, len(parts)) for cid, parts in cluster_parts.items()}
    while True:
        names = {
            cid: " · ".join(p[2] for p in ordered(cluster_parts[cid][: n_used[cid]]))
            or f"Cluster {cid}"
            for cid in masks
        }
        by_name: dict[str, list] = {}
        for cid, name in names.items():
            by_name.setdefault(name, []).append(cid)
        growable = [
            cid
            for cids in by_name.values()
            if len(cids) > 1
            for cid in cids
            if n_used[cid] < len(cluster_parts[cid])
        ]
        if not growable:
            break
        for cid in growable:
            n_used[cid] += 1

    labels = np.full(cluster_ids.shape, noise_label, dtype=object)
    for cid, mask in masks.items():
        labels[mask] = names[cid]
    if not include_details:
        return labels

    details = {
        names[cid]: " · ".join(
            f"{text} {share:.0%} (vs {base:.0%} overall)"
            for _, _, text, share, base in ordered(cluster_parts[cid][: n_used[cid]])
        )
        for cid in masks
    }
    return labels, details


def cluster_color_map(
    map_coords: np.ndarray,
    labels: np.ndarray,
    *,
    noise_label: str = "Unlabelled",
    **palette_kwds,
) -> dict[str, str]:
    """Label -> hex color from cluster positions on a 2-D map.

    Uses datamapplot's position-based palette (hue follows the cluster's angle
    on the map), so different partitions of the same embedding get consistent
    colors, and a 3-D view colored with the same mapping matches the 2-D map.
    """
    try:
        from datamapplot.palette_handling import palette_from_datamap
    except ImportError as exc:
        raise ImportError(
            "datamapplot is required for cluster_color_map (dev dependency group)"
        ) from exc

    labels = np.asarray(labels, dtype=object)
    names = [name for name in np.unique(labels) if name != noise_label]
    locations = np.array([map_coords[labels == name].mean(axis=0) for name in names])
    palette = palette_from_datamap(map_coords, locations, **palette_kwds)
    return dict(zip(names, palette))


def persona_datamap(
    coords: np.ndarray,
    labels: np.ndarray,
    attribute_values: dict[str, Sequence[str]],
    *,
    title: str | None = None,
    sub_title: str | None = None,
    histogram_values: Sequence | None = None,
    **datamapplot_kwds,
):
    """Render a 2-D persona embedding as an interactive deck.gl data map.

    The hover tooltip headlines the first three attributes and lists the rest;
    the search box does a case-insensitive substring match over all of them
    (every attribute value is folded into one hidden string per persona).
    ``histogram_values`` adds a linked histogram that filters visible points
    (e.g. raw ages). Extra datamapplot/render kwargs pass straight through, e.g.
    ``label_color_map`` or a ``colormaps`` dict for an attribute-coloring
    dropdown. Returns a datamapplot interactive figure: it displays inline in
    notebooks and ``.save(path)`` writes a standalone HTML file.
    """
    try:
        import datamapplot
    except ImportError as exc:
        raise ImportError(
            "datamapplot is required for persona_datamap (dev dependency group)"
        ) from exc

    extra = pd.DataFrame(
        {name: [str(v) for v in vals] for name, vals in attribute_values.items()}
    )
    extra["search"] = [" ".join(row) for row in extra.itertuples(index=False)]
    head, tail = list(attribute_values)[:3], list(attribute_values)[3:]
    hover_text = [" · ".join(row) for row in zip(*(extra[name] for name in head))]
    template = (
        "<div style='max-width:280px'><b>{hover_text}</b>"
        + "".join(f"<br>{name}: {{{name}}}" for name in tail)
        + "</div>"
    )

    if histogram_values is not None:
        datamapplot_kwds["histogram_data"] = pd.Series(histogram_values)
    return datamapplot.create_interactive_plot(
        np.asarray(coords),
        np.asarray(labels, dtype=object),
        hover_text=hover_text,
        title=title,
        sub_title=sub_title,
        enable_search=True,
        search_field="search",
        extra_point_data=extra,
        hover_text_html_template=template,
        **datamapplot_kwds,
    )
