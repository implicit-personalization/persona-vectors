import streamlit as st

from src.ui.utils.datasets import load_dataset
from src.ui.utils.helpers import PROMPT_VARIANTS, persona_label, prompt_variant_label
from src.ui.utils.extraction import run_extraction
from src.ui.utils.runtime import cached_model


def render_extract_tab(remote: bool, model_name: str, dataset_source: str) -> None:
    """Render the extraction tab."""

    st.subheader("Extraction")
    st.caption("Run persona-vector extraction using existing src/ modules.")

    if dataset_source == "Local JSONL upload":
        st.file_uploader(
            "personas.jsonl",
            type=["jsonl"],
            key="personas_file",
            help="Expected fields: id, persona, templated_prompt, biography_md",
        )
        st.file_uploader(
            "qa.jsonl",
            type=["jsonl"],
            key="qa_file",
            help="Expected fields: id, qid, type, question, answer, difficulty",
        )

    selected_variants = st.multiselect(
        "Prompt variants",
        options=PROMPT_VARIANTS,
        default=PROMPT_VARIANTS,
        format_func=prompt_variant_label,
        key="extract_prompt_variants",
    )
    if not selected_variants:
        st.info("Select at least one prompt variant.")
        return

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

    selected_persona = st.selectbox(
        "Persona",
        options=personas,
        format_func=persona_label,
        key="extract_persona_select",
    )

    qa_filter_type: str | None
    qa_filter_difficulty: list[int] | None

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        qa_type_select = st.selectbox(
            "QA type",
            options=["all", "explicit", "implicit"],
            index=0,
            key="extract_qa_type_select",
        )
        if qa_type_select == "explicit":
            qa_filter_type = "explicit"
        elif qa_type_select == "implicit":
            qa_filter_type = "implicit"
        else:
            qa_filter_type = None
    with col2:
        difficulty_values = st.multiselect(
            "Difficulty",
            options=[1, 2, 3],
            default=[1, 2, 3],
            key="extract_difficulty_select",
        )
        qa_filter_difficulty = difficulty_values if difficulty_values else None
    with col3:
        total_matching = len(
            dataset.get_qa(
                persona_id=selected_persona.id,
                type=qa_filter_type,
                difficulty=qa_filter_difficulty,
            )
        )
        if total_matching == 0:
            st.warning("No QA pairs match current filters. Update filters to continue.")
            return
        max_questions = st.slider(
            "Max questions",
            min_value=1,
            max_value=total_matching,
            value=total_matching,
            key="extract_max_questions",
        )

    run_clicked = st.button("Run extraction", type="primary")
    if not run_clicked:
        return

    qa_pairs = dataset.get_qa(
        persona_id=selected_persona.id,
        type=qa_filter_type,
        difficulty=qa_filter_difficulty,
    )
    qa_pairs = qa_pairs[: int(max_questions)]

    if not qa_pairs:
        st.error("No QA pairs match current filters")
        return

    status_box = st.empty()
    status_box.info("Extraction in progress...")

    with st.spinner("Loading model..."):
        model = cached_model(model_name=model_name, remote=remote)

    try:
        results = run_extraction(
            model=model,
            model_name=model_name,
            persona=selected_persona,
            qa_pairs=qa_pairs,
            variants=selected_variants,
            remote=remote,
        )
    except Exception as exc:
        st.error(f"Extraction failed: {exc}")
        return

    status_box.success("Extraction complete")
    st.success(f"Saved {len(results)} artifact set(s)")

    for result in results:
        st.markdown(
            f"- `{result.variant}`: `{result.output_dir}` | "
            f"shape=({result.n_questions}, {result.n_layers}, {result.d_model})"
        )
