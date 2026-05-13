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
from persona_vectors.attributes import attribute_color_kwargs
from persona_vectors.hub import parse_vector_config_name
from persona_vectors.plots import (
    build_layered_figure,
    build_pair_similarity_figure,
    build_similarity_figures,
    plot_persona_dendrogram,
    prepare_layered_projection_data,
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

    isomap_2d = build_layered_figure(samples, "isomap", layers=[0], n_components=2)
    assert isomap_2d.data[0].type == "scattergl"

    isomap_graph = build_layered_figure(
        samples,
        "isomap",
        layers=[0],
        n_components=3,
        graph_overlay=True,
        graph_n_neighbors=2,
    )
    assert isomap_graph.data[0].type == "scatter3d"
    assert isomap_graph.data[0].mode == "lines"
    assert isomap_graph.data[1].type == "scatter3d"


def test_high_cardinality_projection_groups_use_one_webgl_trace() -> None:
    samples = _layered_samples(n_samples=45)

    fig = build_layered_figure(samples, "pca", layers=[0], n_components=2)

    assert len(fig.data) == 1
    assert fig.data[0].type == "scattergl"
    assert fig.data[0].name == "Personas"
    assert len(fig.data[0].marker.color) == 45


def test_projection_kmeans_modes_and_numeric_colors() -> None:
    vectors = torch.tensor(
        [
            [[-1.0, -1.0], [-1.0, -1.0]],
            [[-1.0, -0.9], [1.0, 1.0]],
            [[1.0, 1.0], [-1.0, -0.9]],
            [[1.0, 0.9], [1.0, 0.9]],
        ]
    )
    labels = [f"Persona {idx}" for idx in range(4)]
    samples = LayeredSamples(vectors, labels, labels)

    def partitions(frame) -> set[frozenset[str]]:
        return {
            frozenset(trace.text)
            for trace in frame.data
            if trace.text is not None and len(trace.text) > 0
        }

    first_layer = build_layered_figure(
        samples,
        "pca",
        layers=[0, 1],
        n_clusters=2,
        cluster_mode="first_layer",
    )
    assert partitions(first_layer.frames[0]) == partitions(first_layer.frames[1])

    per_layer = build_layered_figure(
        samples,
        "pca",
        layers=[0, 1],
        n_clusters=2,
        cluster_mode="per_layer",
    )
    assert partitions(per_layer.frames[0]) != partitions(per_layer.frames[1])

    numeric = build_layered_figure(
        samples,
        "pca",
        layers=[0],
        color_values=[0.0, 1.0, 2.0, 3.0],
        color_label="Ordinal rank",
        color_tickvals=[0.0, 1.0, 2.0, 3.0],
        color_ticktext=["A", "B", "C", "D"],
    )
    assert numeric.data[0].marker.color == (0.0, 1.0, 2.0, 3.0)
    assert numeric.data[0].marker.colorbar.title.text == "Ordinal rank"


def test_layered_projection_data_can_be_reused_for_color_changes() -> None:
    samples = _layered_samples()
    projection_data = prepare_layered_projection_data(
        samples,
        "pca",
        layers=[0, 1],
    )

    grouped = build_layered_figure(
        samples,
        "pca",
        layers=[0, 1],
        groups=["A", "A", "A", "A", "A"],
        projection_data=projection_data,
    )
    numeric = build_layered_figure(
        samples,
        "pca",
        layers=[0, 1],
        color_values=[0.0, 1.0, 2.0, 3.0, 4.0],
        projection_data=projection_data,
    )

    assert numeric.data[0].x == grouped.data[0].x
    assert [frame.name for frame in numeric.frames] == ["0", "1"]

    with pytest.raises(ValueError, match="projection_data layers"):
        build_layered_figure(
            samples,
            "pca",
            layers=[0],
            projection_data=projection_data,
        )


def test_attribute_color_kwargs_cover_numeric_ordinal_and_nominal() -> None:
    class Dataset:
        attribute_names = ["age", "degree", "city", "wealth"]
        attribute_schema = {
            "persona_fields": {
                "age": {"kind": "numeric"},
                "degree": {
                    "kind": "ordinal",
                    "ordered_values": ["High school", "Bachelor's", "Graduate"],
                },
                "city": {"kind": "nominal"},
                "wealth": {
                    "kind": "ordinal",
                    "ordered_values": ["Less than $5,000", "$5,000 to $20,000"],
                },
            }
        }

        def attribute_values(self, name: str, persona_ids: list[str]) -> list[object]:
            values = {
                "age": [20, 40, 60],
                "degree": ["High school", "Graduate", "Bachelor's"],
                "city": ["A", "B", "C"],
                "wealth": ["Less than $5,000", "$5,000 to $20,000"],
            }
            return values[name][: len(persona_ids)]

    dataset = Dataset()
    persona_ids = ["p1", "p2", "p3"]

    numeric = attribute_color_kwargs(dataset, "age", persona_ids)
    assert numeric["color_values"] == [20.0, 40.0, 60.0]
    assert numeric["colorscale"] == "Viridis"

    ordinal = attribute_color_kwargs(dataset, "degree", persona_ids)
    assert ordinal["color_values"] == [0.0, 2.0, 1.0]
    assert ordinal["color_ticktext"] == ["High school", "Bachelor's", "Graduate"]

    wealth = attribute_color_kwargs(dataset, "wealth", persona_ids[:2])
    assert wealth["color_values"] == [0.0, 1.0]
    assert wealth["color_ticktext"] == [
        "Less than &#36;5,000",
        "&#36;5,000 to &#36;20,000",
    ]

    nominal = attribute_color_kwargs(dataset, "city", persona_ids, max_categories=2)
    assert nominal["groups"] == ["A", "B", "Other"]


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
        assert store.available_variants([]) == []
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
        hub_store._datasets = {
            "templated": [
                {"vector": vectors.tolist()},
                {"vector": (vectors + 1).tolist()},
            ],
            "biography": [
                {"vector": vectors.tolist()},
            ],
        }
        hub_store._index = {
            "templated": {"persona-001": 0, "persona-002": 1},
            "biography": {"persona-001": 0},
        }
        hub_store._names = {
            "templated": {
                "persona-001": "Test Persona",
                "persona-002": "Other Persona",
            },
            "biography": {"persona-001": "Test Persona"},
        }
        hub_store._metadata_complete.update({"templated", "biography"})
        assert hub_store.list_personas(["templated"]) == [
            "persona-001",
            "persona-002",
        ]
        assert hub_store.list_personas(["templated", "biography"]) == ["persona-001"]
        assert hub_store.list_personas([]) == []
        assert hub_store.available_variants([]) == []
        assert hub_store.persona_names(["persona-001"], variants=[]) == {}
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


def test_hf_metadata_stops_after_requested_personas() -> None:
    class CountingDataset(list):
        def __init__(self, rows):
            super().__init__(rows)
            self.rows_read = 0

        def select_columns(self, columns):
            source = self

            class Selected:
                def __len__(self):
                    return len(source)

                def __getitem__(self, i: int):
                    source.rows_read += 1
                    return {column: source[i][column] for column in columns}

            return Selected()

    vector = torch.zeros(2, 3)
    dataset = CountingDataset(
        [
            {
                "persona_id": f"persona-{idx:03d}",
                "name": f"Persona {idx}",
                "vector": vector.tolist(),
            }
            for idx in range(20)
        ]
    )
    store = HFActivationStore("test/repo", "test/model")
    store._datasets["templated"] = dataset

    assert store.persona_names(
        ["persona-001", "persona-003"], variants=["templated"]
    ) == {
        "persona-001": "Persona 1",
        "persona-003": "Persona 3",
    }
    assert dataset.rows_read == 4

    assert torch.allclose(store.load("templated", "persona-001"), vector)
    assert dataset.rows_read == 4

    # Asking for a not-yet-seen id resumes from the last scanned row instead
    # of restarting at 0.
    assert store.persona_names(["persona-007"], variants=["templated"]) == {
        "persona-007": "Persona 7",
    }
    assert dataset.rows_read == 8

    # A full listing finishes the scan; subsequent calls are cached.
    assert len(store.list_personas(["templated"])) == 20
    assert dataset.rows_read == 20
    assert store.list_personas(["templated"])[:2] == ["persona-000", "persona-001"]
    assert dataset.rows_read == 20
