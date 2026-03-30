from src.synth_persona_io import PersonaData

DATASET_SOURCES = ["HuggingFace: synth-persona", "Local JSONL upload"]
PROMPT_VARIANTS = ["templated", "biography"]
ANALYSIS_MODES = ["Cosine similarity", "PCA", "UMAP"]

PROMPT_VARIANT_LABELS = {
    "templated": "Template",
    "biography": "Biography",
}

SYSTEM_PROMPT_OPTIONS = [
    ("empty", "None"),
    ("templated", "Template"),
    ("biography", "Biography"),
]


def prompt_variant_label(variant: str) -> str:
    """Return a human-friendly prompt-variant label."""

    return PROMPT_VARIANT_LABELS.get(variant, variant.title())


def persona_label(persona: PersonaData) -> str:
    """Format a persona for selection widgets."""

    return f"{persona.name} ({persona.id})"


def persona_display_label(persona_id: str, persona_name: str | None) -> str:
    """Format a persona id with an optional display name."""

    if persona_name:
        return f"{persona_name} ({persona_id})"
    return persona_id
