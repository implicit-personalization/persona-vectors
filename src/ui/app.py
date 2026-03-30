import streamlit as st
import torch

from src.environment import load_env, set_seed
from src.ui.tabs.chat import render_chat_tab
from src.ui.tabs.extract import render_extract_tab
from src.ui.tabs.load_compare import render_load_compare_tab
from src.ui.utils.helpers import DATASET_SOURCES
from src.ui.utils.runtime import list_remote_models

DEFAULT_MODEL = "google/gemma-2-2b-it"
REMOTE_DEFAULT_MODEL = "google/gemma-2-9b-it"


def _sidebar_model_controls() -> tuple[bool, str]:
    with st.sidebar:
        st.header("Settings")
        remote = st.toggle("Remote (NDIF)", value=True)

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


def _sidebar_shared_controls() -> tuple[bool, str, str]:
    remote, model_name = _sidebar_model_controls()

    with st.sidebar:
        st.divider()
        st.subheader("Data")
        dataset_source = st.selectbox("Dataset source", DATASET_SOURCES)

    return remote, model_name, dataset_source


def main() -> None:
    load_env()
    torch.set_grad_enabled(False)
    set_seed(1337)

    st.set_page_config(page_title="Persona Vectors", page_icon="🧭", layout="wide")
    st.title("Persona Vectors Monitor")

    remote, model_name, dataset_source = _sidebar_shared_controls()

    tab_extract, tab_load, tab_chat = st.tabs(["Extract", "Load + Compare", "Chat"])
    with tab_extract:
        render_extract_tab(remote, model_name, dataset_source)
    with tab_load:
        render_load_compare_tab(model_name)
    with tab_chat:
        render_chat_tab(remote, model_name, dataset_source)


if __name__ == "__main__":
    main()
