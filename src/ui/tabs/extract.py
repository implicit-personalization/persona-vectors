import streamlit as st

from src.ui.utils.datasets import load_dataset
from src.ui.utils.extraction import run_extraction
from src.ui.utils.helpers import (
    PROMPT_VARIANTS,
    persona_label,
    prompt_variant_label,
    widget_key,
)
from src.ui.utils.runtime import cached_model


def _extract_widget_key(
    model_name: str, remote: bool, dataset_source: str, suffix: str
) -> str:
    return widget_key("extract", str(remote), model_name, dataset_source, suffix)


def _render_local_dataset_uploads() -> None:
    """Render file inputs for local dataset uploads."""

    with st.expander("Local dataset upload", expanded=True):
        st.file_uploader(
            "personas.jsonl",
            type=["jsonl"],
            key="extract__personas_file",
            help="Expected fields: id, persona, templated_prompt, biography_md",
        )
        st.file_uploader(
            "qa.jsonl",
            type=["jsonl"],
            key="extract__qa_file",
            help="Expected fields: id, qid, type, question, answer, difficulty",
        )


def render_extract_tab(remote: bool, model_name: str, dataset_source: str) -> None:
    """Render the extraction tab."""

    st.title("Extract")

    if dataset_source == "Local JSONL upload":
        _render_local_dataset_uploads()

    selected_variants = st.multiselect(
        "Prompt variants",
        options=PROMPT_VARIANTS,
        default=PROMPT_VARIANTS,
        format_func=prompt_variant_label,
        key=_extract_widget_key(model_name, remote, dataset_source, "prompt_variants"),
    )
    if not selected_variants:
        st.info("Select at least one prompt variant.")
        return

    try:
        dataset, dataset_status = load_dataset(dataset_source)
        st.caption(dataset_status)
    except Exception as exc:
        st.error(f"Could not load data: {exc}")
        st.info(
            "Upload both JSONL files or switch to the built-in SynthPersona source."
        )
        return

    personas = list(dataset)
    if not personas:
        st.warning("No personas found in the selected dataset.")
        st.info(
            "Try another dataset source or check that the personas file is not empty."
        )
        return

    selected_persona = st.selectbox(
        "Persona",
        options=personas,
        format_func=persona_label,
        key=_extract_widget_key(model_name, remote, dataset_source, "persona_select"),
    )

    st.caption("Filters")

    qa_filter_type: str | None
    qa_filter_difficulty: list[int] | None

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        qa_type_select = st.selectbox(
            "QA type",
            options=["all", "explicit", "implicit"],
            index=0,
            key=_extract_widget_key(
                model_name, remote, dataset_source, "qa_type_select"
            ),
        )
        qa_filter_type = (
            qa_type_select if qa_type_select in ("explicit", "implicit") else None
        )
    with col2:
        difficulty_values = st.multiselect(
            "Difficulty",
            options=[1, 2, 3],
            default=[1, 2, 3],
            key=_extract_widget_key(
                model_name, remote, dataset_source, "difficulty_select"
            ),
        )
        qa_filter_difficulty = difficulty_values if difficulty_values else None

    all_qa_pairs = dataset.get_qa(
        persona_id=selected_persona.id,
        type=qa_filter_type,
        difficulty=qa_filter_difficulty,
    )
    if not all_qa_pairs:
        st.warning("No QA pairs match the current filters.")
        st.info("Widen the filters or reset them to continue.")
        return

    with col3:
        max_questions = st.slider(
            "Max questions",
            min_value=1,
            max_value=len(all_qa_pairs),
            value=len(all_qa_pairs),
            key=_extract_widget_key(
                model_name, remote, dataset_source, "max_questions"
            ),
        )

    run_clicked = st.button("Run extraction", type="primary")
    if not run_clicked:
        return

    qa_pairs = all_qa_pairs[:max_questions]

    status_box = st.empty()
    status_box.info("Extraction in progress...")
    progress = st.progress(0, text="Preparing extraction...")

    with st.spinner("Loading model..."):
        model = cached_model(model_name=model_name, remote=remote)

    try:
        total_steps = len(selected_variants)
        results = []

        for idx, variant in enumerate(selected_variants, start=1):
            progress.progress(
                (idx - 1) / total_steps if total_steps else 1.0,
                text=f"Processing variant {idx}/{total_steps}: {variant}",
            )
            variant_results = run_extraction(
                model=model,
                model_name=model_name,
                persona=selected_persona,
                qa_pairs=qa_pairs,
                variants=[variant],
                remote=remote,
            )
            results.extend(variant_results)

        progress.progress(1.0, text="Extraction complete")
    except Exception as exc:
        st.error(f"Extraction failed: {exc}")
        return
    finally:
        progress.empty()

    status_box.success("Extraction complete")
    st.success(f"Saved {len(results)} artifact set(s)")

    for result in results:
        st.markdown(
            f"- {prompt_variant_label(result.variant)}: {result.n_questions} questions, "
            f"{result.n_layers} layers, {result.d_model} hidden size"
        )
