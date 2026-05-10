import json
import tempfile

import pytest
import torch
from persona_data.synth_persona import BASELINE_PERSONA_ID, BASELINE_PERSONA_NAME

import persona_vectors  # noqa: F401
from persona_vectors.analysis import (
    LayeredSamples,
    list_comparison_personas,
    load_persona_vectors,
    load_variant_vectors,
    run_saved_activation_analysis,
)
from persona_vectors.artifacts import (
    DEFAULT_MASK_STRATEGY,
    ActivationStore,
    HFActivationStore,
    activation_config_name,
    discover_activation_models,
    model_dir_name,
)
from persona_vectors.hub import parse_vector_config_name
from persona_vectors.plots import (
    build_layered_figure,
    build_pair_similarity_figure,
    build_similarity_figures,
    plot_persona_dendrogram,
)
from persona_vectors.steering import compute_steering_vector


def _layered_samples(n_samples: int = 5) -> LayeredSamples:
    values = torch.arange(n_samples * 2 * 8, dtype=torch.float32).reshape(
        n_samples, 2, 8
    )
    labels = [f"Persona {idx}" for idx in range(n_samples)]
    return LayeredSamples(values, labels, labels)


def test_projection_plots_cover_2d_and_3d() -> None:
    samples = _layered_samples()

    pca_3d = build_layered_figure(samples, "pca", layers=[0, 1], n_components=3)
    assert pca_3d.data[0].type == "scatter3d"

    umap_2d = build_layered_figure(samples, "umap", layers=[0], n_components=2)
    assert umap_2d.data[0].type == "scattergl"

    umap_3d = build_layered_figure(samples, "umap", layers=[0], n_components=3)
    assert umap_3d.data[0].type == "scatter3d"


def test_projection_plots_validate_component_count() -> None:
    samples = _layered_samples(n_samples=2)

    with pytest.raises(ValueError, match="n_components=3"):
        build_layered_figure(samples, "pca", layers=[0], n_components=3)

    with pytest.raises(ValueError, match="UMAP requires at least 3 samples"):
        build_layered_figure(samples, "umap", layers=[0], n_components=2)


def test_layered_dendrogram_has_layer_controls() -> None:
    samples = _layered_samples()

    fig = plot_persona_dendrogram(samples, layered=True, layers=[0, 1])

    assert [frame.name for frame in fig.frames] == ["0", "1"]
    assert fig.layout.sliders[0].steps[0].label == "0"
    assert fig.layout.sliders[0].pad.t == 115
    assert fig.layout.updatemenus[0].buttons[0].label == "Play"
    assert fig.layout.margin.b == 260
    assert all(frame.layout.yaxis.range is not None for frame in fig.frames)
    assert fig.frames[0].layout.yaxis.range == fig.frames[1].layout.yaxis.range


@pytest.mark.parametrize("linkage", ["average", "complete", "single", "ward"])
def test_dendrogram_supports_linkage_options(linkage: str) -> None:
    samples = _layered_samples()

    fig = plot_persona_dendrogram(samples, linkage=linkage)

    assert fig.data
    assert linkage.title() in fig.layout.title.text


def test_smoke() -> None:
    print("✓ imports OK")

    with tempfile.TemporaryDirectory() as tmp:
        store = ActivationStore(
            "test/model", root_dir=tmp, mask_strategy="answer_previous"
        )
        vectors = torch.arange(4 * 8, dtype=torch.float32).reshape(4, 8)
        sample_ids = ["q0", "q1", "q2"]

        saved_dir = store.save(
            "templated",
            "persona-001",
            "Test Persona",
            vectors,
            sample_ids,
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

        loaded_vectors = store.load("templated", "persona-001")
        assert torch.allclose(loaded_vectors, vectors)
        assert store.list_personas(["templated"]) == ["persona-001"]
        assert store.list_personas([]) == []
        assert store.available_variants(["templated", "biography"]) == ["templated"]
        assert store.available_variants([]) == ["templated"]
        assert store.persona_names(["persona-001"], variants=["templated"]) == {
            "persona-001": "Test Persona"
        }
        assert store.list_layers(["templated"], ["persona-001"]) == [0, 1, 2, 3]
        assert discover_activation_models(tmp, "answer_previous") == ["test/model"]
        assert discover_activation_models(tmp, "answer_mean") == []

        assert DEFAULT_MASK_STRATEGY == "answer_mean"
        assert model_dir_name("org/model") == "org__model"
        assert activation_config_name("org/model", "answer_mean") == (
            "org__model__answer_mean"
        )
        assert parse_vector_config_name("org__model__answer_mean") == (
            "org/model",
            "answer_mean",
        )
        assert parse_vector_config_name("not-a-vector-config") is None

        hub_store = HFActivationStore("test/repo", "test/model")
        assert hub_store.config_name == "test__model__answer_mean"
        hub_store._cache = {
            "templated": {
                "persona-001": {
                    "name": "Test Persona",
                    "vector": vectors.tolist(),
                },
                "persona-002": {
                    "name": "Other Persona",
                    "vector": (vectors + 1).tolist(),
                },
            },
            "biography": {
                "persona-001": {
                    "name": "Test Persona",
                    "vector": vectors.tolist(),
                }
            },
        }
        assert hub_store.list_personas(["templated"]) == [
            "persona-001",
            "persona-002",
        ]
        assert hub_store.list_personas(["templated", "biography"]) == ["persona-001"]
        assert hub_store.list_personas([]) == ["persona-001"]
        assert hub_store.persona_names(["persona-001"], variants=[]) == {
            "persona-001": "Test Persona"
        }
        assert hub_store.persona_names(["persona-001"], variants=["templated"]) == {
            "persona-001": "Test Persona"
        }
        assert hub_store.list_layers(["templated"], ["persona-001"]) == [0, 1, 2, 3]
        assert hub_store.list_layers(["templated", "biography"], ["persona-001"]) == [
            0,
            1,
            2,
            3,
        ]
        assert hub_store.list_layers(["templated"], ["missing"]) == []
        assert torch.allclose(hub_store.load("templated", "persona-001"), vectors)
        for call in (
            lambda: hub_store.load(
                "templated", "persona-001", mask_strategy="answer_previous"
            ),
            lambda: hub_store.list_personas(
                ["templated"], mask_strategy="answer_previous"
            ),
        ):
            try:
                call()
            except ValueError:
                continue
            raise AssertionError("HFActivationStore should reject mismatched masks")

        store.save(
            "templated",
            BASELINE_PERSONA_ID,
            BASELINE_PERSONA_NAME,
            vectors,
            sample_ids,
        )
        assert store.list_personas(["templated"]) == [
            BASELINE_PERSONA_ID,
            "persona-001",
        ]
        assert list_comparison_personas(store, ["templated"]) == ["persona-001"]
        assert list_comparison_personas(
            store, ["templated"], include_baseline=True
        ) == [BASELINE_PERSONA_ID, "persona-001"]

    with tempfile.TemporaryDirectory() as tmp:
        store = ActivationStore("test/model", root_dir=tmp)
        sample_ids = ["q0", "q1", "q2"]
        for idx, persona_id in enumerate(["persona-001", "persona-002"]):
            vectors = torch.arange(4 * 8, dtype=torch.float32).reshape(4, 8) + idx * 100
            store.save(
                "biography",
                persona_id,
                f"Test Persona {idx + 1}",
                vectors,
                sample_ids,
            )

        pm = load_persona_vectors(store, "biography")
        assert pm.vectors.shape == (2, 4, 8)
        vm = load_variant_vectors(
            store,
            ["biography"],
            persona_ids=["persona-001", "persona-002"],
        )
        assert vm["biography"].vectors.shape == (2, 4, 8)
        assert vm["biography"].labels == ["Test Persona 1", "Test Persona 2"]

        build_layered_figure(pm, "similarity", layers=[0, 1])
        build_pair_similarity_figure(pm, layers=[0, 1])
        similarity_fig, pair_fig = build_similarity_figures(pm, layers=[0, 1])
        assert similarity_fig.data[0].type == "heatmap"
        assert pair_fig.data

        outputs = run_saved_activation_analysis(
            model_name="test/model",
            activations_dir=tmp,
            output_dir=tmp,
            variant="biography",
            mask_strategy="answer_mean",
            layers=[0, 1],
        )
        assert {
            "persona_vector_pca",
            "persona_vector_similarity",
            "persona_pair_similarity",
            "pca_scree",
        } <= outputs.keys()
        assert all(path.exists() for path in outputs.values())

    with tempfile.TemporaryDirectory() as tmp:
        store = ActivationStore("test/model", root_dir=tmp)
        sample_ids = ["q0", "q1"]
        templated = torch.zeros(3, 4)
        biography = torch.ones(3, 4)
        store.save(
            "templated",
            "persona-001",
            "Test Persona",
            templated,
            sample_ids,
            mask_strategy="answer_mean",
        )
        store.save(
            "biography",
            "persona-001",
            "Test Persona",
            biography,
            sample_ids,
            mask_strategy="answer_mean",
        )

        sv = compute_steering_vector(
            "persona-001",
            "test/model",
            layer_idx=1,
            mask_strategy="answer_mean",
            activations_dir=tmp,
            verbose=False,
        )
        assert sv["steering_vector"].shape == (1, 1, 4)
        assert torch.allclose(sv["steering_vector"], torch.ones(1, 1, 4))

    print("✓ smoke test passed")
