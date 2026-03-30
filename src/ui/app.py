from pathlib import Path
from tempfile import mkdtemp
from typing import Literal

import streamlit as st
import torch

from src.activation_io import load_per_question_vectors
from src.environment import get_artifacts_dir, load_env, set_seed
from src.plots import plot_multiple_layer_similarities
from src.synth_persona_io import SynthPersonaDataset
from src.ui.utils.extraction import run_extraction
from src.ui.utils.local_dataset import LocalPersonaDataset
from src.ui.utils.runtime import list_remote_models, load_model

# NOTE: This has a lot of code some function in the future will be abastracted away elsewhre and this could also be splitted into different files
# But for now it is mostly a proof of concept and can be modified accordingly to fit your needs

load_env()
torch.set_grad_enabled(False)
set_seed(1337)

DEFAULT_MODEL = "google/gemma-2-2b-it"
REMOTE_DEFAULT_MODEL = "google/gemma-2-9b-it"


@st.cache_resource(show_spinner=False)
def cached_model(model_name: str, remote: bool):
    return load_model(model_name=model_name, remote=remote)


@st.cache_resource(show_spinner=False)
def cached_hf_dataset() -> SynthPersonaDataset:
    return SynthPersonaDataset()


def _upload_cache_dir() -> Path:
    cache_dir = st.session_state.get("_upload_cache_dir")
    if cache_dir is None:
        cache_dir = mkdtemp(prefix="persona_vectors_uploads_")
        st.session_state["_upload_cache_dir"] = cache_dir
    return Path(cache_dir)


def _uploaded_file_to_temp_path(uploaded_file, stem: str) -> Path:
    suffix = Path(uploaded_file.name).suffix or ".jsonl"
    temp_path = _upload_cache_dir() / f"{stem}{suffix}"
    temp_path.write_bytes(uploaded_file.getvalue())
    return temp_path


def _load_dataset(dataset_source: str):
    if dataset_source == "HuggingFace: synth-persona":
        return cached_hf_dataset(), "Loaded HF dataset"

    personas_file = st.session_state.get("personas_file")
    qa_file = st.session_state.get("qa_file")
    if personas_file is None or qa_file is None:
        raise ValueError("Upload both personas.jsonl and qa.jsonl files")

    personas_path = _uploaded_file_to_temp_path(personas_file, stem="personas")
    qa_path = _uploaded_file_to_temp_path(qa_file, stem="qa")
    return (
        LocalPersonaDataset(personas_path=personas_path, qa_path=qa_path),
        "Loaded local dataset",
    )


def _persona_label(persona) -> str:
    return f"{persona.name} ({persona.id})"


def _model_dir_name(model_name: str) -> str:
    return model_name.replace("/", "__")


def _display_model_name(model_dir_name: str) -> str:
    return model_dir_name.replace("__", "/")


def _sidebar_model_controls() -> tuple[bool, str]:
    with st.sidebar:
        st.header("Settings")
        remote = st.toggle("Remote (NDIF)", value=False)

        if remote:
            remote_models = list_remote_models()
            if remote_models:
                default_model = (
                    REMOTE_DEFAULT_MODEL
                    if REMOTE_DEFAULT_MODEL in remote_models
                    else remote_models[0]
                )
                model_name = st.selectbox(
                    "Remote model",
                    options=remote_models,
                    index=remote_models.index(default_model),
                )
                st.caption("Using running NDIF models.")
            else:
                st.error("No running NDIF models found.")
                model_name = REMOTE_DEFAULT_MODEL
        else:
            model_name = st.text_input("HuggingFace model", value=DEFAULT_MODEL)

    return remote, model_name


def _sidebar_shared_controls() -> tuple[bool, str, str, str]:
    remote, model_name = _sidebar_model_controls()

    with st.sidebar:
        st.divider()
        st.subheader("Extraction")
        dataset_source = st.selectbox(
            "Dataset source",
            ["HuggingFace: synth-persona", "Local JSONL upload"],
        )
        variants = st.multiselect(
            "Prompt variants",
            options=["templated", "biography"],
            default=["templated", "biography"],
        )

    return remote, model_name, dataset_source, variants


def _list_available_models(artifacts_root: str | Path) -> list[str]:
    root = Path(artifacts_root)
    if not root.exists():
        return []
    return sorted([_display_model_name(d.name) for d in root.iterdir() if d.is_dir()])


def _list_available_personas(
    artifacts_root: str | Path, model_name: str, variant: str
) -> list[str]:
    model_dir = Path(artifacts_root) / _model_dir_name(model_name) / variant
    if not model_dir.exists():
        return []
    return sorted([d.name for d in model_dir.iterdir() if d.is_dir()])


def _run_extraction_tab(
    remote: bool, model_name: str, dataset_source: str, variants: list[str]
) -> None:
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

    try:
        dataset, dataset_status = _load_dataset(dataset_source)
        st.success(dataset_status)
    except Exception as exc:
        st.warning(str(exc))
        return

    personas = list(dataset)
    if not personas:
        st.error("No personas found in selected dataset")
        return

    selected_label = st.selectbox(
        "Persona",
        options=[_persona_label(persona) for persona in personas],
    )
    selected_persona = personas[
        [_persona_label(persona) for persona in personas].index(selected_label)
    ]

    qa_filter_type: Literal["explicit", "implicit"] | None
    qa_filter_difficulty: list[int] | None

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        qa_type_select = st.selectbox(
            "QA type", options=["all", "explicit", "implicit"], index=0
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
        )

    if not variants:
        st.info("Select at least one prompt variant.")
        return

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
            variants=variants,
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


def _run_load_tab(model_name: str) -> None:
    st.subheader("Load + Compare")
    st.caption("Load saved vectors and compare layer-wise cosine similarity.")

    artifacts_root = st.text_input(
        "Artifacts root",
        value=str(get_artifacts_dir() / "activations"),
    )

    available_models = _list_available_models(artifacts_root)
    if not available_models:
        st.info("No models found. Run extraction first to populate the directory.")

    col1, col2 = st.columns(2)
    with col1:
        variant_a = st.selectbox(
            "Variant A", options=["templated", "biography"], index=0
        )
    with col2:
        variant_b = st.selectbox(
            "Variant B", options=["templated", "biography"], index=1
        )

    available_personas_a = _list_available_personas(
        artifacts_root, model_name, variant_a
    )
    available_personas_b = _list_available_personas(
        artifacts_root, model_name, variant_b
    )
    available_personas = sorted(set(available_personas_a + available_personas_b))

    if available_personas:
        persona_options = available_personas
    else:
        persona_options = []

    persona_ids = st.multiselect(
        "Persona ids",
        options=persona_options,
        default=persona_options[:1] if len(persona_options) > 1 else persona_options,
    )
    if not available_personas:
        st.info("No personas found for this model. Run extraction first.")

    if st.button("Load and compare"):
        if not persona_ids:
            st.error("Select at least one persona")
            return

        traces: list[tuple[str, torch.Tensor, torch.Tensor]] = []
        errors: list[str] = []
        for persona_id in persona_ids:
            try:
                vectors_a, _ = load_per_question_vectors(
                    root_dir=artifacts_root,
                    model_name=model_name,
                    prompt_variant=variant_a,
                    persona_id=persona_id,
                )
                vectors_b, _ = load_per_question_vectors(
                    root_dir=artifacts_root,
                    model_name=model_name,
                    prompt_variant=variant_b,
                    persona_id=persona_id,
                )
            except Exception as exc:
                errors.append(f"{persona_id}: {exc}")
                continue

            mean_a = vectors_a.mean(dim=0)
            mean_b = vectors_b.mean(dim=0)
            traces.append((persona_id, mean_a, mean_b))

        if errors:
            for err in errors:
                st.error(f"Failed to load vectors: {err}")
        if not traces:
            st.error("No personas loaded successfully")
            return

        fig = plot_multiple_layer_similarities(
            traces,
            title=f"{variant_a} vs {variant_b}",
            show=False,
        )
        st.plotly_chart(fig, width="stretch")
        shape_msgs = ", ".join(f"{t[0]}={tuple(t[1].shape)}" for t in traces)
        st.success(f"Loaded: {shape_msgs}")


def main() -> None:
    st.set_page_config(page_title="Persona Vectors", page_icon="🧭", layout="wide")
    st.title("Persona Vectors Monitor")

    remote, model_name, dataset_source, variants = _sidebar_shared_controls()

    tab_extract, tab_load = st.tabs(["Extract", "Load + Compare"])
    with tab_extract:
        _run_extraction_tab(remote, model_name, dataset_source, variants)
    with tab_load:
        _run_load_tab(model_name)


if __name__ == "__main__":
    main()
