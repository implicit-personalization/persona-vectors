import streamlit as st

from src.ui.state import chat_session_key, get_chat_state, reset_chat_state
from src.ui.utils.chat import ChatReply, generate_chat_reply, resolve_system_prompt
from src.ui.utils.datasets import load_dataset
from src.ui.utils.helpers import (
    MODE_LABEL_TO_KEY,
    MODE_LABELS,
    VARIANT_LABELS,
    persona_label,
    widget_key,
)
from src.ui.utils.runtime import cached_model


def _render_chat_message(message: dict[str, str]) -> None:
    if not message.get("content"):
        return
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


def _chat_widget_key(context_key: str, suffix: str) -> str:
    return widget_key(context_key, suffix)


def _set_pending_chat_prompt(prompt_key: str, prompt: str) -> None:
    st.session_state[prompt_key] = prompt


def _mark_chat_started(chat_started_key: str) -> None:
    st.session_state[chat_started_key] = True


def _consume_pending_chat_prompt(prompt_key: str) -> str | None:
    return st.session_state.pop(prompt_key, None)


def _clear_chat_ui_state(*keys: str) -> None:
    for key in keys:
        st.session_state.pop(key, None)


def _render_chat_suggestions(
    dataset: object,
    selected_persona_id: str,
    context_key: str,
    pending_prompt_key: str,
) -> None:
    """Render clickable question suggestions for the selected persona."""

    explicit_pairs = dataset.get_qa(selected_persona_id, type="explicit")
    implicit_pairs = dataset.get_qa(selected_persona_id, type="implicit")

    if not explicit_pairs and not implicit_pairs:
        return

    eq_col, iq_col = st.columns(2)
    with eq_col:
        st.caption("Explicit")
        if explicit_pairs:
            st.button(
                explicit_pairs[0].question,
                key=_chat_widget_key(context_key, "suggest_explicit"),
                on_click=_set_pending_chat_prompt,
                args=(pending_prompt_key, explicit_pairs[0].question),
            )
    with iq_col:
        st.caption("Implicit")
        if implicit_pairs:
            st.button(
                implicit_pairs[0].question,
                key=_chat_widget_key(context_key, "suggest_implicit"),
                on_click=_set_pending_chat_prompt,
                args=(pending_prompt_key, implicit_pairs[0].question),
            )


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
        st.error("No personas found in the selected dataset.")
        return

    context_key = chat_session_key(model_name, remote, dataset_source)
    chat_state = get_chat_state(model_name, remote, dataset_source)

    col1, col2 = st.columns([2, 1])
    with col1:
        selected_index = next(
            (i for i, p in enumerate(personas) if p.id == chat_state["persona_id"]),
            0,
        )

        selected_persona = st.selectbox(
            "Persona",
            options=personas,
            index=selected_index,
            format_func=persona_label,
            key=_chat_widget_key(context_key, "persona_select"),
        )
    with col2:
        current_mode_label = VARIANT_LABELS.get(chat_state["prompt_mode"], "None")
        prompt_mode_label = st.selectbox(
            "System prompt",
            options=MODE_LABELS,
            index=MODE_LABELS.index(current_mode_label),
            key=_chat_widget_key(context_key, "system_prompt_select"),
        )
        prompt_mode = MODE_LABEL_TO_KEY[prompt_mode_label]

    max_new_tokens = st.slider(
        "Max new tokens",
        min_value=16,
        max_value=512,
        value=256,
        step=16,
        key=_chat_widget_key(context_key, "max_new_tokens"),
    )

    advanced_generation = st.toggle(
        "Advanced generation",
        value=False,
        key=_chat_widget_key(context_key, "advanced_generation"),
    )

    if advanced_generation:
        use_sampling = st.checkbox(
            "Random sampling",
            value=False,
            key=_chat_widget_key(context_key, "use_sampling"),
        )

        st.caption(
            "Temperature, top-p, and top-k only apply when random sampling is on."
        )

        sampling_disabled = not use_sampling
        sampling_col1, sampling_col2, sampling_col3 = st.columns(3)
        with sampling_col1:
            temperature = st.slider(
                "Temperature",
                min_value=0.01,
                max_value=2.0,
                value=1.0,
                step=0.01,
                disabled=sampling_disabled,
                key=_chat_widget_key(context_key, "temperature"),
            )
        with sampling_col2:
            top_p = st.slider(
                "Top-p",
                min_value=0.01,
                max_value=1.0,
                value=1.0,
                step=0.01,
                disabled=sampling_disabled,
                key=_chat_widget_key(context_key, "top_p"),
            )
        with sampling_col3:
            top_k = st.slider(
                "Top-k (0 = off)",
                min_value=0,
                max_value=100,
                value=50,
                step=1,
                disabled=sampling_disabled,
                key=_chat_widget_key(context_key, "top_k"),
            )

        config_col1, config_col2 = st.columns([2, 1])
        with config_col1:
            repetition_penalty = st.slider(
                "Repetition penalty",
                min_value=0.5,
                max_value=2.0,
                value=1.0,
                step=0.05,
                key=_chat_widget_key(context_key, "repetition_penalty"),
            )
        with config_col2:
            seed_disabled = sampling_disabled or remote
            seed_enabled = st.checkbox(
                "Fix seed",
                value=False,
                disabled=seed_disabled,
                key=_chat_widget_key(context_key, "seed_enabled"),
            )
            if seed_enabled:
                seed = int(
                    st.number_input(
                        "Seed",
                        min_value=0,
                        max_value=2_147_483_647,
                        value=0,
                        step=1,
                        disabled=seed_disabled,
                        key=_chat_widget_key(context_key, "seed"),
                    )
                )
            else:
                seed = None

        if remote:
            st.caption("Seed is local-only and disabled for remote runs.")
    else:
        use_sampling = False
        temperature = 1.0
        top_p = 1.0
        top_k = 50
        repetition_penalty = 1.0
        seed = None

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
        reset_chat_state(model_name, remote, dataset_source)
        _clear_chat_ui_state(
            _chat_widget_key(context_key, "pending_prompt"),
            _chat_widget_key(context_key, "chat_input"),
            _chat_widget_key(context_key, "chat_started"),
        )
        if had_history:
            st.info("Chat history reset because the persona or system prompt changed.")

    with st.expander("Active system prompt"):
        st.code(active_system_prompt or "", language="text")

    if st.button("Reset chat"):
        reset_chat_state(model_name, remote, dataset_source)
        _clear_chat_ui_state(
            _chat_widget_key(context_key, "pending_prompt"),
            _chat_widget_key(context_key, "chat_input"),
            _chat_widget_key(context_key, "chat_started"),
        )
        st.rerun()

    pending_prompt_key = _chat_widget_key(context_key, "pending_prompt")
    chat_input_key = _chat_widget_key(context_key, "chat_input")
    chat_started_key = _chat_widget_key(context_key, "chat_started")

    chat_log = st.container()

    with chat_log:
        for message in chat_state["messages"]:
            _render_chat_message(message)

        # Show suggestions only before the conversation starts.
        show_suggestions = not (
            chat_state["messages"]
            or st.session_state.get(chat_started_key)
            or pending_prompt_key in st.session_state
        )
        if show_suggestions:
            _render_chat_suggestions(
                dataset=dataset,
                selected_persona_id=selected_persona.id,
                context_key=context_key,
                pending_prompt_key=pending_prompt_key,
            )

    # Keep the input at the bottom so the conversation and suggestions render above it.
    user_prompt = st.chat_input(
        "Ask something...",
        key=chat_input_key,
        on_submit=_mark_chat_started,
        args=(chat_started_key,),
    )
    if not user_prompt:
        user_prompt = _consume_pending_chat_prompt(pending_prompt_key)

    if not user_prompt:
        return

    st.session_state[chat_started_key] = True

    chat_state["messages"].append({"role": "user", "content": user_prompt})
    with chat_log:
        _render_chat_message(chat_state["messages"][-1])

    messages = []
    if active_system_prompt:
        messages.append({"role": "system", "content": active_system_prompt})
    messages.extend(chat_state["messages"])

    with st.spinner("Generating reply..."):
        model = cached_model(model_name=model_name, remote=remote)
        do_sample = bool(advanced_generation and use_sampling)
        generation_seed = (
            seed if do_sample and seed is not None and not remote else None
        )
        try:
            reply: ChatReply = generate_chat_reply(
                model=model,
                messages=messages,
                remote=remote,
                past_key_values=chat_state["past_key_values"],
                max_new_tokens=int(max_new_tokens),
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                seed=generation_seed,
            )
        except Exception as exc:
            with chat_log:
                st.error(f"Chat generation failed: `{exc}`")
            chat_state["messages"].pop()
            return

    chat_state["messages"].append({"role": "assistant", "content": reply.text})
    chat_state["past_key_values"] = reply.past_key_values if not remote else None
    with chat_log:
        _render_chat_message(chat_state["messages"][-1])
        st.caption(
            f"Generated {reply.output_tokens} tokens from {reply.prompt_tokens} prompt tokens."
        )
