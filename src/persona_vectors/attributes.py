from collections import Counter
from collections.abc import Iterable
from typing import Any

DEFAULT_MAX_ATTRIBUTE_CATEGORIES = 12


def _plotly_tick_label(value: object) -> str:
    """Return a literal-safe label for Plotly tick text."""

    return str(value).replace("$", "&#36;")


def attribute_schema(persona_dataset: Any) -> dict[str, dict[str, Any]]:
    """Return the persona-field schema from a SynthPersonaDataset-like object."""

    return persona_dataset.attribute_schema.get("persona_fields", {})


def attribute_display_label(persona_dataset: Any, attribute_name: str) -> str:
    """Format an attribute label with kind and cardinality metadata."""

    info = attribute_schema(persona_dataset).get(attribute_name)
    if not info:
        return attribute_name
    unique = info.get("n_unique_seed_values")
    suffix = f", {unique} values" if unique else ""
    return f"{attribute_name} ({info.get('kind', 'unknown')}{suffix})"


def categorical_attribute_labels(
    values: Iterable[object],
    max_categories: int = DEFAULT_MAX_ATTRIBUTE_CATEGORIES,
) -> list[str]:
    """Return categorical labels, collapsing infrequent values to ``Other``."""

    values = list(values)
    counts = Counter(values)
    keep = {
        value
        for value, _ in sorted(
            counts.items(), key=lambda item: (-item[1], str(item[0]))
        )[:max_categories]
    }
    return [str(value) if value in keep else "Other" for value in values]


def attribute_color_kwargs(
    persona_dataset: Any,
    attribute_name: str,
    persona_ids: list[str],
    *,
    max_categories: int = DEFAULT_MAX_ATTRIBUTE_CATEGORIES,
    numeric_colorscale: str = "Viridis",
    ordinal_colorscale: str = "Plasma",
) -> dict[str, object]:
    """Return ``build_layered_figure`` color kwargs for a persona attribute."""

    schema = attribute_schema(persona_dataset)
    info = schema[attribute_name]
    values = list(persona_dataset.attribute_values(attribute_name, persona_ids))
    kind = info["kind"]

    if kind == "numeric":
        return {
            "color_values": [float(value) for value in values],
            "color_label": attribute_name,
            "colorscale": numeric_colorscale,
        }
    if kind == "ordinal":
        ordered = info["ordered_values"]
        ranks = {value: idx for idx, value in enumerate(ordered)}
        return {
            "color_values": [float(ranks[value]) for value in values],
            "color_label": attribute_name,
            "colorscale": ordinal_colorscale,
            "color_tickvals": [float(idx) for idx in range(len(ordered))],
            "color_ticktext": [_plotly_tick_label(value) for value in ordered],
        }
    return {"groups": categorical_attribute_labels(values, max_categories)}
