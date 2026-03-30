from pathlib import Path
from tempfile import mkdtemp
from typing import Any

import streamlit as st

from src.synth_persona_io import SynthPersonaDataset

from .local_dataset import LocalPersonaDataset


@st.cache_resource(show_spinner=False)
def cached_hf_dataset() -> SynthPersonaDataset:
    """Load the default SynthPersona HuggingFace dataset once."""

    return SynthPersonaDataset()


def _upload_cache_dir() -> Path:
    cache_dir = st.session_state.get("_upload_cache_dir")
    if cache_dir is None:
        cache_dir = mkdtemp(prefix="persona_vectors_uploads_")
        st.session_state["_upload_cache_dir"] = cache_dir
    return Path(cache_dir)


def _uploaded_file_to_temp_path(uploaded_file: Any, stem: str) -> Path:
    suffix = Path(uploaded_file.name).suffix or ".jsonl"
    temp_path = _upload_cache_dir() / f"{stem}{suffix}"
    temp_path.write_bytes(uploaded_file.getvalue())
    return temp_path


def load_dataset(
    dataset_source: str,
) -> tuple[SynthPersonaDataset | LocalPersonaDataset, str]:
    """Load the selected dataset source for the UI."""

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
