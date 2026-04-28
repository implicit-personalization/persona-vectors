import json
import tempfile

import torch

import persona_vectors  # noqa: F401
from persona_vectors.artifacts import (
    DEFAULT_MASK_STRATEGY,
    ActivationStore,
    list_personas,
    model_dir_name,
)


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

print("✓ smoke test passed")
