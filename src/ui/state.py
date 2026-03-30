import streamlit as st


def chat_session_key(model_name: str, remote: bool, dataset_source: str) -> str:
    """Build the session-state key for a chat context."""

    return f"chat_state::{remote}::{model_name}::{dataset_source}"


def _default_chat_state() -> dict[str, object]:
    return {
        "messages": [],
        "persona_id": None,
        "prompt_mode": "empty",
        "max_new_tokens": 256,
        "past_key_values": None,
    }


def get_chat_state(
    model_name: str, remote: bool, dataset_source: str
) -> dict[str, object]:
    """Return the mutable chat state for the active context."""

    key = chat_session_key(model_name, remote, dataset_source)
    state = st.session_state.get(key)
    if state is None:
        state = _default_chat_state()
        st.session_state[key] = state
    return state


def reset_chat_state(model_name: str, remote: bool, dataset_source: str) -> None:
    """Reset chat history and cache for the active context."""

    state = get_chat_state(model_name, remote, dataset_source)
    state["messages"] = []
    state["past_key_values"] = None
