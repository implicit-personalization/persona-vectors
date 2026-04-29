import json
import tempfile

import torch
from persona_data.prompts import BASELINE_PERSONA_ID, BASELINE_PERSONA_NAME

import persona_vectors  # noqa: F401
from persona_vectors.analysis import (
    load_persona_mean_samples,
    run_saved_activation_analysis,
)
from persona_vectors.artifacts import (
    DEFAULT_MASK_STRATEGY,
    ActivationStore,
    list_personas,
    model_dir_name,
)
from persona_vectors.plots import build_layered_figure

print("✓ imports OK")

with tempfile.TemporaryDirectory() as tmp:
    store = ActivationStore("test/model", root_dir=tmp)
    vectors = torch.arange(3 * 4 * 8, dtype=torch.float32).reshape(3, 4, 8)
    sample_ids = ["q0", "q1", "q2"]

    saved_dir = store.save(
        "templated",
        "persona-001",
        "Test Persona",
        vectors,
        sample_ids,
        mask_strategy="answer_previous",
    )
    expected_dir = store.root_dir / "test__model" / "answer_previous" / "templated"
    assert saved_dir == expected_dir
    assert (expected_dir / "manifest.json").exists()
    assert (expected_dir / "persona-001.safetensors").exists()
    manifest = json.loads((saved_dir / "manifest.json").read_text())
    assert manifest["personas"]["persona-001"] == {
        "name": "Test Persona",
        "sample_ids": sample_ids,
    }

    loaded_vectors, loaded_sample_ids = store.load(
        "templated", "persona-001", mask_strategy="answer_previous"
    )
    assert torch.allclose(loaded_vectors, vectors)
    assert loaded_sample_ids == sample_ids
    assert list_personas(
        tmp, "test/model", ["templated"], mask_strategy="answer_previous"
    ) == ["persona-001"]

    assert DEFAULT_MASK_STRATEGY == "answer_mean"
    assert model_dir_name("org/model") == "org__model"

    store.save(
        "baseline",
        BASELINE_PERSONA_ID,
        BASELINE_PERSONA_NAME,
        vectors,
        sample_ids,
        mask_strategy="answer_previous",
    )
    other_vectors = vectors + 1
    store.save(
        "baseline",
        "stale-persona-id",
        "Stale Persona",
        other_vectors,
        sample_ids,
        mask_strategy="answer_previous",
    )
    assert list_personas(
        tmp, "test/model", ["baseline"], mask_strategy="answer_previous"
    ) == [BASELINE_PERSONA_ID]

with tempfile.TemporaryDirectory() as tmp:
    store = ActivationStore("test/model", root_dir=tmp)
    sample_ids = ["q0", "q1", "q2"]
    for idx, persona_id in enumerate(["persona-001", "persona-002"]):
        vectors = (
            torch.arange(3 * 4 * 8, dtype=torch.float32).reshape(3, 4, 8) + idx * 100
        )
        store.save(
            "biography",
            persona_id,
            f"Test Persona {idx + 1}",
            vectors,
            sample_ids,
            mask_strategy="answer_mean",
        )

    pm = load_persona_mean_samples(tmp, "test/model", "biography", "answer_mean")
    assert pm.vectors.shape == (2, 4, 8)

    build_layered_figure(pm, "similarity", layers=[0, 1])

    outputs = run_saved_activation_analysis(
        model_name="test/model",
        activations_dir=tmp,
        output_dir=tmp,
        variant="biography",
        mask_strategy="answer_mean",
        layers=[0, 1],
    )
    assert {
        "persona_mean_pca",
        "persona_mean_similarity",
    } <= outputs.keys()
    assert all(path.exists() for path in outputs.values())

print("✓ smoke test passed")
