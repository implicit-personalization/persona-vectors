import nnsight
import streamlit as st
from nnsight import LanguageModel


@st.cache_data(show_spinner=False, ttl=30)
def list_remote_models() -> list[str]:
    """Return the NDIF language models that are currently running."""

    try:
        status = nnsight.ndif_status()
    except Exception:
        return []

    model_names: list[str] = []

    for entry in status.values():
        if not isinstance(entry, dict):
            continue
        if entry.get("model_class") != "LanguageModel":
            continue

        state = entry.get("state")
        state_name = getattr(state, "name", None) or getattr(state, "value", None)
        if state_name != "RUNNING":
            continue

        repo_id = entry.get("repo_id")
        if isinstance(repo_id, str):
            model_names.append(repo_id)

    return sorted(dict.fromkeys(model_names))


@st.cache_resource(show_spinner=False)
def cached_model(model_name: str, remote: bool) -> LanguageModel:
    """Load an nnsight model with resource caching."""

    return load_model(model_name=model_name, remote=remote)


def load_model(model_name: str, remote: bool) -> LanguageModel:
    """Load an nnsight model for local or remote tracing."""
    if remote:
        return LanguageModel(model_name)
    return LanguageModel(model_name, dtype="auto", device_map="auto", dispatch=True)
