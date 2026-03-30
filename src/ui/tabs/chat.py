import streamlit as st

from src.ui.state import get_chat_state, reset_chat_state
from src.ui.utils.chat import ChatReply, generate_chat_reply, resolve_system_prompt
from src.ui.utils.datasets import load_dataset
from src.ui.utils.helpers import SYSTEM_PROMPT_OPTIONS, persona_label
from src.ui.utils.runtime import cached_model


def _render_chat_message(message: dict[str, str]) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"] or "")


def render_chat_tab(remote: bool, model_name: str, dataset_source: str) -> None:
    """Render the chat tab."""

    st.subheader("Chat")
    st.caption(
        "Multi-turn chat with model generation and persona-based system prompts."
    )

    try:
        dataset, dataset_status = load_dataset(dataset_source)
        st.success(dataset_status)
    except Exception as exc:
        st.warning(str(exc))
        return

    personas = list(dataset)
    if not personas:
        st.error("No personas found in selected dataset")
        return

    chat_state = get_chat_state(model_name, remote, dataset_source)

    col1, col2 = st.columns([2, 1])
    with col1:
        selected_index = 0
        if chat_state["persona_id"] is not None:
            for idx, persona in enumerate(personas):
                if persona.id == chat_state["persona_id"]:
                    selected_index = idx
                    break

        selected_persona = st.selectbox(
            "Persona",
            options=personas,
            index=selected_index,
            format_func=persona_label,
            key="chat_persona_select",
        )
    with col2:
        mode_labels = [label for _, label in SYSTEM_PROMPT_OPTIONS]
        mode_keys = {label: key for key, label in SYSTEM_PROMPT_OPTIONS}
        current_mode_label = next(
            (
                label
                for key, label in SYSTEM_PROMPT_OPTIONS
                if key == chat_state["prompt_mode"]
            ),
            "None",
        )
        prompt_mode_label = st.selectbox(
            "System prompt",
            options=mode_labels,
            index=mode_labels.index(current_mode_label),
            key="chat_system_prompt_select",
        )
        prompt_mode = mode_keys[prompt_mode_label]

    max_new_tokens = st.slider(
        "Max new tokens",
        min_value=16,
        max_value=512,
        value=int(chat_state["max_new_tokens"]),
        step=16,
        key="chat_max_new_tokens",
    )

    active_system_prompt = resolve_system_prompt(
        persona=selected_persona,
        mode=prompt_mode,
    )

    changed_context = (
        chat_state["persona_id"] != selected_persona.id
        or chat_state["prompt_mode"] != prompt_mode
    )
    if changed_context:
        had_history = bool(chat_state["messages"])
        chat_state["persona_id"] = selected_persona.id
        chat_state["prompt_mode"] = prompt_mode
        chat_state["max_new_tokens"] = int(max_new_tokens)
        reset_chat_state(model_name, remote, dataset_source)
        if had_history:
            st.info("Chat history reset because the persona or system prompt changed.")
    else:
        chat_state["max_new_tokens"] = int(max_new_tokens)

    st.markdown("**Active system prompt**")
    st.code(active_system_prompt or "", language="text")

    if st.button("Reset chat"):
        reset_chat_state(model_name, remote, dataset_source)
        st.rerun()

    for message in chat_state["messages"]:
        _render_chat_message(message)

    user_prompt = st.chat_input("Ask something...")
    if not user_prompt:
        return

    chat_state["messages"].append({"role": "user", "content": user_prompt})
    _render_chat_message(chat_state["messages"][-1])

    messages = []
    if active_system_prompt:
        messages.append({"role": "system", "content": active_system_prompt})
    messages.extend(chat_state["messages"])

    with st.spinner("Generating reply..."):
        model = cached_model(model_name=model_name, remote=remote)
        try:
            reply: ChatReply = generate_chat_reply(
                model=model,
                messages=messages,
                remote=remote,
                past_key_values=chat_state["past_key_values"],
                max_new_tokens=int(max_new_tokens),
            )
        except Exception as exc:
            st.error(f"Chat generation failed: {exc}")
            chat_state["messages"].pop()
            return

    chat_state["messages"].append({"role": "assistant", "content": reply.text})
    chat_state["past_key_values"] = reply.past_key_values if not remote else None
    _render_chat_message(chat_state["messages"][-1])

    st.caption(
        f"Generated {reply.output_tokens} tokens from {reply.prompt_tokens} prompt tokens."
    )
