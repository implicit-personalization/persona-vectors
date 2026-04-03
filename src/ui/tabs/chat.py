import json

import streamlit as st
import streamlit.components.v1 as components

from src.ui.state import chat_session_key, get_chat_state, reset_chat_state
from src.ui.utils.chat import ChatReply, generate_chat_reply, resolve_system_prompt
from src.ui.utils.chat_export import save_chat_export
from src.ui.utils.datasets import load_dataset
from src.ui.utils.helpers import (
    MODE_LABEL_TO_KEY,
    MODE_LABELS,
    VARIANT_LABELS,
    persona_label,
    widget_key,
)
from src.ui.utils.runtime import cached_model

_VISIBLE_MESSAGE_COUNT = 5


def _render_chat_message(message: dict[str, str]) -> None:
    if not message.get("content"):
        return
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


def _mark_chat_started(chat_started_key: str) -> None:
    st.session_state[chat_started_key] = True


def _clear_chat_ui_state(*keys: str) -> None:
    for key in keys:
        st.session_state.pop(key, None)


def _dataset_caption(dataset_source: str, dataset_status: str) -> str:
    return dataset_status


def _render_copy_button(prompt_text: str, button_key: str) -> None:
    button_id = f"copy-prompt-{button_key}"
    button_label = "Copy prompt"
    components.html(
        f"""
        <button id={json.dumps(button_id)} style="
            width: 100%;
            padding: 0.4rem 0.75rem;
            border-radius: 0.5rem;
            border: 1px solid rgba(49, 51, 63, 0.14);
            background: #f8f9fa;
            color: #1f2937;
            font: inherit;
            cursor: pointer;
        ">{button_label}</button>
        <script>
        const button = document.getElementById({json.dumps(button_id)});
        const promptText = {json.dumps(prompt_text)};
        const defaultLabel = {json.dumps(button_label)};
        button.addEventListener("click", async () => {{
          try {{
            await navigator.clipboard.writeText(promptText);
            button.textContent = "Copied";
          }} catch (error) {{
            button.textContent = "Copy failed";
          }}
          window.setTimeout(() => {{ button.textContent = defaultLabel; }}, 1200);
        }});
        </script>
        """,
        height=42,
    )


def render_chat_tab(remote: bool, model_name: str, dataset_source: str) -> None:
    """Render the chat tab."""

    st.title("Chat")

    context_key = chat_session_key(model_name, dataset_source)
    chat_state = get_chat_state(model_name, remote, dataset_source)
    try:
        dataset, dataset_status = load_dataset(dataset_source)
        st.caption(_dataset_caption(dataset_source, dataset_status))
    except Exception as exc:
        st.error(f"Could not load data: {exc}")
        st.info("Check the selected dataset source or upload both JSONL files.")
        return

    personas = list(dataset)
    persona_select_key = widget_key(context_key, "persona_select")
    prompt_mode_select_key = widget_key(context_key, "system_prompt_select")

    if not personas:
        st.warning("No personas found in the selected dataset.")
        st.info("Try a different dataset source or upload a non-empty personas file.")
        return

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
            key=persona_select_key,
        )
    with col2:
        current_mode_label = VARIANT_LABELS.get(chat_state["prompt_mode"], "None")
        prompt_mode_label = st.selectbox(
            "Prompt",
            options=MODE_LABELS,
            index=MODE_LABELS.index(current_mode_label),
            key=prompt_mode_select_key,
        )
        prompt_mode = MODE_LABEL_TO_KEY[prompt_mode_label]

    advanced_generation = st.toggle(
        "Advanced generation",
        value=False,
        key=widget_key(context_key, "advanced_generation"),
    )

    if advanced_generation:
        max_new_tokens = st.slider(
            "Max new tokens",
            min_value=16,
            max_value=512,
            value=256,
            step=16,
            key=widget_key(context_key, "max_new_tokens"),
        )

        use_sampling = st.checkbox(
            "Random sampling",
            value=False,
            key=widget_key(context_key, "use_sampling"),
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
                key=widget_key(context_key, "temperature"),
            )
        with sampling_col2:
            top_p = st.slider(
                "Top-p",
                min_value=0.01,
                max_value=1.0,
                value=1.0,
                step=0.01,
                disabled=sampling_disabled,
                key=widget_key(context_key, "top_p"),
            )
        with sampling_col3:
            top_k = st.slider(
                "Top-k (0 = off)",
                min_value=0,
                max_value=100,
                value=50,
                step=1,
                disabled=sampling_disabled,
                key=widget_key(context_key, "top_k"),
            )

        config_col1, config_col2 = st.columns([2, 1])
        with config_col1:
            repetition_penalty = st.slider(
                "Repetition penalty",
                min_value=0.5,
                max_value=2.0,
                value=1.0,
                step=0.05,
                key=widget_key(context_key, "repetition_penalty"),
            )
        with config_col2:
            seed_disabled = sampling_disabled or remote
            seed_enabled = st.checkbox(
                "Fix seed",
                value=False,
                disabled=seed_disabled,
                key=widget_key(context_key, "seed_enabled"),
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
                        key=widget_key(context_key, "seed"),
                    )
                )
            else:
                seed = None

        if remote:
            st.caption("Seed is local-only and disabled for remote runs.")
    else:
        max_new_tokens = 256
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

    chat_input_key = widget_key(context_key, "chat_input")
    chat_started_key = widget_key(context_key, "chat_started")
    show_all_key = widget_key(context_key, "show_all_messages")
    custom_prompt_key = widget_key(context_key, "custom_system_prompt")
    export_success_message: str | None = None

    action_col1, action_col2, action_col3 = st.columns(3)
    with action_col1:
        if active_system_prompt:
            _render_copy_button(
                active_system_prompt, widget_key(context_key, "copy_prompt")
            )
        else:
            st.button("Copy prompt", disabled=True, use_container_width=True)
    with action_col2:
        if st.button("Export chat", use_container_width=True):
            export_path = save_chat_export(
                model_name=model_name,
                dataset_source=dataset_source,
                persona_id=selected_persona.id,
                persona_name=getattr(selected_persona, "name", None),
                prompt_mode=prompt_mode,
                system_prompt=active_system_prompt,
                messages=chat_state["messages"],
                generation={
                    "max_new_tokens": int(max_new_tokens),
                    "advanced_generation": bool(advanced_generation),
                    "use_sampling": bool(use_sampling),
                    "temperature": float(temperature),
                    "top_p": float(top_p),
                    "top_k": int(top_k),
                    "repetition_penalty": float(repetition_penalty),
                    "seed": seed,
                },
            )
            export_success_message = f"Saved chat export to {export_path}"
    with action_col3:
        if st.button("Reset chat", use_container_width=True):
            reset_chat_state(model_name, remote, dataset_source)
            _clear_chat_ui_state(
                chat_input_key,
                chat_started_key,
                show_all_key,
                custom_prompt_key,
            )
            st.rerun()

    if export_success_message:
        st.success(export_success_message)

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
            chat_input_key,
            chat_started_key,
            show_all_key,
            custom_prompt_key,
        )
        if had_history:
            st.info("Chat history reset because the persona or system prompt changed.")

    chat_log = st.container()

    with chat_log:
        # System prompt as first item in conversation — collapsed by default, editable.
        if prompt_mode != "empty":
            if custom_prompt_key not in st.session_state:
                st.session_state[custom_prompt_key] = active_system_prompt
            with st.expander("Edit prompt", expanded=False):
                active_system_prompt = (
                    st.text_area(
                        "Prompt",
                        key=custom_prompt_key,
                        height=200,
                        label_visibility="collapsed",
                    )
                    or None
                )

        # Collapse older messages, show only the most recent ones.
        messages = chat_state["messages"]
        if len(messages) > _VISIBLE_MESSAGE_COUNT and not st.session_state.get(
            show_all_key, False
        ):
            hidden_count = len(messages) - _VISIBLE_MESSAGE_COUNT
            if st.button(
                f"Show earlier messages ({hidden_count} hidden)",
                key=widget_key(context_key, "show_all_btn"),
            ):
                st.session_state[show_all_key] = True
                st.rerun()
            visible_messages = messages[-_VISIBLE_MESSAGE_COUNT:]
        else:
            visible_messages = messages

        for message in visible_messages:
            _render_chat_message(message)

    user_prompt = st.chat_input(
        "Ask something...",
        key=chat_input_key,
        on_submit=_mark_chat_started,
        args=(chat_started_key,),
    )

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
                st.error(f"Could not generate a reply: {exc}")
                st.info("Try a shorter prompt, reset the chat, or switch personas.")
            chat_state["messages"].pop()
            return

    chat_state["messages"].append({"role": "assistant", "content": reply.text})
    chat_state["past_key_values"] = reply.past_key_values if not remote else None

    export_path = save_chat_export(
        model_name=model_name,
        dataset_source=dataset_source,
        persona_id=selected_persona.id,
        persona_name=getattr(selected_persona, "name", None),
        prompt_mode=prompt_mode,
        system_prompt=active_system_prompt,
        messages=chat_state["messages"],
        generation={
            "max_new_tokens": int(max_new_tokens),
            "advanced_generation": bool(advanced_generation),
            "use_sampling": bool(use_sampling),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "top_k": int(top_k),
            "repetition_penalty": float(repetition_penalty),
            "seed": generation_seed,
        },
    )

    with chat_log:
        _render_chat_message(chat_state["messages"][-1])
        with st.expander("Generation details", expanded=False):
            st.caption(
                f"{reply.output_tokens} output tokens, {reply.prompt_tokens} prompt tokens."
            )
