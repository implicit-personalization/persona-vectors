from src.synth_persona_io import PersonaData

# Variant key -> human-readable label mapping
VARIANT_LABELS = {"empty": "None", "templated": "Template", "biography": "Biography"}

# Variants that correspond to actual system prompts (excludes "empty")
PROMPT_VARIANTS = ["templated", "biography"]

# For selectbox options: list of labels in definition order
MODE_LABELS = list(VARIANT_LABELS.values())

# Reverse lookup: label -> key
MODE_LABEL_TO_KEY = {v: k for k, v in VARIANT_LABELS.items()}

DATASET_SOURCES = ["HuggingFace: synth-persona", "Local JSONL upload"]
ANALYSIS_MODES = ["Cosine similarity", "PCA", "UMAP"]

ANALYSIS_LABELS = {
    "PCA": ("PCA", "PC1", "PC2"),
    "UMAP": ("UMAP", "UMAP 1", "UMAP 2"),
}


def widget_key(*parts: str) -> str:
    """Generate a namespaced Streamlit widget key from parts."""

    return "::".join(parts)


def prompt_variant_label(variant: str) -> str:
    """Return a human-friendly prompt-variant label."""

    return VARIANT_LABELS.get(variant, variant.title())


def persona_label(persona: PersonaData) -> str:
    """Format a persona for selection widgets."""

    return f"{persona.name} ({persona.id})"


def persona_display_label(persona_id: str, persona_name: str | None) -> str:
    """Format a persona id with an optional display name."""

    if persona_name:
        return f"{persona_name} ({persona_id})"
    return persona_id
