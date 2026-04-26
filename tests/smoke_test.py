"""
Smoke test for the persona-vectors package.

Verifies that:
1. All public modules import cleanly (catches missing files in the dist).
2. Core logic works end-to-end without a GPU or loaded model.

Run via:
  uv run --isolated --no-project --with dist/*.whl tests/smoke_test.py
"""

import tempfile
from pathlib import Path

import torch

import persona_vectors  # noqa: F401 – package-level import
from persona_vectors.analysis import (
    build_embedding_figure,
    pairwise_cosine_similarity,
    pca_explained_variance,
    project_pca,
)
from persona_vectors.artifacts import (
    ActivationStore,
    list_layers,
    list_personas,
    load_mean_activations,
    load_persona_names,
    model_dir_name,
)
from persona_vectors.extraction import (
    ExtractionResult,
    MaskStrategy,
    PromptSpans,
    Span,
    _build_mask,
)
from persona_vectors.plots import (
    plot_scree,
    plot_similarity_matrix,
    plot_similarity_matrix_grid,
)

# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------

print("✓ All imports OK")

# ---------------------------------------------------------------------------
# 2. Helpers
# ---------------------------------------------------------------------------

N_QUESTIONS = 5
NUM_LAYERS = 4
HIDDEN_SIZE = 8
MODEL_NAME = "test/model"
PERSONA_ID = "persona-001"
PERSONA_NAME = "Test Persona"


def make_vectors() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(N_QUESTIONS, NUM_LAYERS, HIDDEN_SIZE)


# ---------------------------------------------------------------------------
# 3. ActivationStore: save / load roundtrip
# ---------------------------------------------------------------------------

with tempfile.TemporaryDirectory() as tmp:
    store = ActivationStore(MODEL_NAME, root_dir=tmp)
    vectors = make_vectors()
    questions = [f"Q{i}" for i in range(N_QUESTIONS)]

    saved_dir = store.save("templated", PERSONA_ID, PERSONA_NAME, vectors, questions)
    assert saved_dir.exists(), "artifact dir not created"
    assert (saved_dir / "activations.safetensors").exists(), "tensors not saved"
    assert (saved_dir / "metadata.json").exists(), "metadata not saved"

    loaded_vectors, loaded_questions = store.load("templated", PERSONA_ID)
    assert loaded_vectors.shape == vectors.shape, "shape mismatch after load"
    assert torch.allclose(loaded_vectors, vectors), (
        "tensor values changed after roundtrip"
    )
    assert loaded_questions == questions, "questions changed after roundtrip"

    print("✓ ActivationStore save/load roundtrip OK")

    # list_personas
    personas = list_personas(tmp, MODEL_NAME, ["templated"])
    assert PERSONA_ID in personas, "persona not found by list_personas"
    print("✓ list_personas OK")

    # load_persona_names
    names = load_persona_names(tmp, MODEL_NAME, ["templated"], [PERSONA_ID])
    assert names.get(PERSONA_ID) == PERSONA_NAME, "persona name not found"
    print("✓ load_persona_names OK")

    # list_layers
    layers = list_layers(tmp, MODEL_NAME, ["templated"], [PERSONA_ID])
    assert layers == list(range(NUM_LAYERS)), f"unexpected layers: {layers}"
    print("✓ list_layers OK")

# ---------------------------------------------------------------------------
# 4. ActivationStore: error cases
# ---------------------------------------------------------------------------

with tempfile.TemporaryDirectory() as tmp:
    store = ActivationStore(MODEL_NAME, root_dir=tmp)

    try:
        store.load("templated", "nonexistent")
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass

    try:
        store.save("templated", PERSONA_ID, PERSONA_NAME, torch.zeros(3, 4), ["q"])
        assert False, "expected ValueError for wrong ndim"
    except ValueError:
        pass

    try:
        store.save(
            "templated",
            PERSONA_ID,
            PERSONA_NAME,
            make_vectors(),
            ["only_one_question"],
        )
        assert False, "expected ValueError for question count mismatch"
    except ValueError:
        pass

    print("✓ ActivationStore error handling OK")

# ---------------------------------------------------------------------------
# 5. load_mean_activations roundtrip
# ---------------------------------------------------------------------------

with tempfile.TemporaryDirectory() as tmp:
    store = ActivationStore(MODEL_NAME, root_dir=tmp)
    vectors = make_vectors()
    questions = [f"Q{i}" for i in range(N_QUESTIONS)]
    store.save("biography", PERSONA_ID, PERSONA_NAME, vectors, questions)
    store.save("templated", PERSONA_ID, PERSONA_NAME, vectors, questions)

    traces, names, errors = load_mean_activations(
        tmp, MODEL_NAME, [PERSONA_ID], "biography", "templated"
    )
    assert not errors, f"unexpected errors: {errors}"
    assert len(traces) == 1, "expected one trace"
    pid, mean_a, mean_b = traces[0]
    assert pid == PERSONA_ID
    assert mean_a.shape == (NUM_LAYERS, HIDDEN_SIZE)
    assert mean_b.shape == (NUM_LAYERS, HIDDEN_SIZE)
    print("✓ load_mean_activations OK")

# ---------------------------------------------------------------------------
# 6. model_dir_name
# ---------------------------------------------------------------------------

assert model_dir_name("org/model") == "org__model"
assert model_dir_name("plain") == "plain"
print("✓ model_dir_name OK")

# ---------------------------------------------------------------------------
# 7. project_pca
# ---------------------------------------------------------------------------

torch.manual_seed(1)
samples = torch.randn(10, 16)
projected = project_pca(samples)
assert projected.shape == (10, 2), f"unexpected PCA shape: {projected.shape}"

try:
    project_pca(torch.randn(4))
    assert False, "expected ValueError for 1D input"
except ValueError:
    pass

print("✓ project_pca OK")

# ---------------------------------------------------------------------------
# 8. build_embedding_figure
# ---------------------------------------------------------------------------

coords = torch.randn(6, 2)
labels = ["A", "A", "B", "B", "C", "C"]
hover = ["h0", "h1", "h2", "h3", "h4", "h5"]
fig = build_embedding_figure(coords, labels, "Test", "x", "y", hover_text=hover)
assert fig is not None
assert len(fig.data) == 3, "expected one trace per unique label"

# without hover
fig2 = build_embedding_figure(coords, labels, "Test", "x", "y")
assert fig2 is not None

try:
    build_embedding_figure(torch.randn(6, 3), labels, "T", "x", "y")
    assert False, "expected ValueError for wrong coord shape"
except ValueError:
    pass

try:
    build_embedding_figure(coords, labels[:4], "T", "x", "y")
    assert False, "expected ValueError for label count mismatch"
except ValueError:
    pass

print("✓ build_embedding_figure OK")

# ---------------------------------------------------------------------------
# 9. pairwise_cosine_similarity
# ---------------------------------------------------------------------------

vectors = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]), torch.tensor([1.0, 1.0])]
similarity = pairwise_cosine_similarity(vectors)
assert similarity.shape == (3, 3)
assert torch.isclose(similarity[0, 0], torch.tensor(1.0))

# centered variant: subtracting the mean changes the off-diagonals
sim_centered = pairwise_cosine_similarity(vectors, center=True)
assert sim_centered.shape == (3, 3)
assert not torch.allclose(similarity, sim_centered)

try:
    pairwise_cosine_similarity([])
    assert False, "expected ValueError for empty vectors"
except ValueError:
    pass

print("✓ pairwise_cosine_similarity OK")

# ---------------------------------------------------------------------------
# 10. plot_similarity_matrix helpers
# ---------------------------------------------------------------------------

small_matrix = torch.tensor([[1.0, -0.25], [-0.25, 1.0]])
single_fig = plot_similarity_matrix(small_matrix, ["A", "B"])
assert len(single_fig.data) == 1

grid_fig = plot_similarity_matrix_grid(
    [small_matrix, small_matrix, small_matrix, small_matrix],
    ["A", "B"],
    ["L1", "L2", "L3", "L4"],
)
assert len(grid_fig.data) == 4

print("✓ plot_similarity_matrix helpers OK")

# ---------------------------------------------------------------------------
# 11. persona mask strategies
# ---------------------------------------------------------------------------

persona_spans = PromptSpans(
    template=Span(0, 3, 0, 3),
    question=Span(3, 5, 3, 5),
    response=Span(5, 7, 5, 7),
)
input_ids = torch.tensor([10, 11, 12, 13, 14, 15, 16])

mask = _build_mask(
    len(input_ids), persona_spans, MaskStrategy.PERSONA_MEAN, input_ids, set()
)
assert mask.tolist() == [True, True, True, False, False, False, False]

mask = _build_mask(
    len(input_ids), persona_spans, MaskStrategy.PERSONA_LAST, input_ids, set()
)
assert mask.tolist() == [False, False, True, False, False, False, False]

print("✓ persona mask strategies OK")

# ---------------------------------------------------------------------------
# 12. ExtractionResult dataclass
# ---------------------------------------------------------------------------

result = ExtractionResult(
    variant="biography",
    output_dir=Path("/tmp/test"),
    n_questions=10,
    persona_name="Ada Lovelace",
)
assert result.variant == "biography"
assert result.n_questions == 10
print("✓ ExtractionResult OK")

# ---------------------------------------------------------------------------
# 13. pca_explained_variance
# ---------------------------------------------------------------------------

torch.manual_seed(2)
samples = torch.randn(20, 12)
ratios = pca_explained_variance(samples, n_components=5)
assert ratios.shape == (5,)
assert ratios.sum() <= 1.0 + 1e-6
assert (ratios >= 0).all()

# full-rank default
ratios_full = pca_explained_variance(samples)
assert ratios_full.shape == (min(samples.shape),)

try:
    pca_explained_variance(torch.randn(4))
    assert False, "expected ValueError for 1D input"
except ValueError:
    pass

print("✓ pca_explained_variance OK")

fig_scree = plot_scree({"a": ratios, "b": ratios})
assert len(fig_scree.data) > 0

print("✓ new plot helpers OK")

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------

print("\nAll smoke tests passed.")
