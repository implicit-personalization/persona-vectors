import json
import sys
import tempfile
import types

import numpy as np
import pytest
import torch
from nnterp import StandardizedTransformer
from persona_data.synth_persona import BASELINE_PERSONA_ID, BASELINE_PERSONA_NAME
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from transformers import GPT2Config, GPT2LMHeadModel

import persona_vectors  # noqa: F401
from persona_vectors import analysis as analysis_module
from persona_vectors.activations import extract_activations
from persona_vectors.analysis import (
    AnalysisDataset,
    LayeredSamples,
    load_analysis_dataset,
    load_persona_vectors,
    load_variant_vectors,
    pca_explained_variance,
    prepare_cluster_samples,
    project_pca,
    project_umap,
    run_saved_activation_analysis,
)
from persona_vectors.artifacts import (
    DEFAULT_MASK_STRATEGY,
    HFPersonaVectorStore,
    PersonaVectorStore,
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
    plot_attribute_layer_selectivity_heatmap,
    plot_persona_dendrogram,
    prepare_kmeans_groups,
    prepare_layered_projection_data,
)
from persona_vectors.probes import (
    AttributeLabels,
    evaluate_regression,
    load_probe_artifact,
    make_linear_probe,
    save_probe_artifact,
)
from persona_vectors.steering import compute_steering_vector


def _layered_samples(n_samples: int = 5) -> LayeredSamples:
    values = torch.arange(n_samples * 2 * 8, dtype=torch.float32).reshape(
        n_samples, 2, 8
    )
    labels = [f"Persona {idx}" for idx in range(n_samples)]
    return LayeredSamples(values, labels, labels)


def test_extract_activations_returns_masked_layer_means() -> None:
    torch.manual_seed(0)
    base = GPT2LMHeadModel(
        GPT2Config(n_layer=2, n_head=2, n_embd=8, n_positions=16, vocab_size=32)
    ).eval()
    model = StandardizedTransformer(base, check_renaming=False, allow_dispatch=False)

    input_ids = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    token_mask = torch.tensor([False, True, True, False])

    activations = extract_activations(model, [input_ids], [token_mask], remote=False)

    assert isinstance(activations, torch.Tensor)
    assert activations.shape == (2, 8)
    assert torch.isfinite(activations).all()


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


def test_pca_and_umap_projection_preprocessing_normalizes_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vectors = torch.tensor(
        [
            [2.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 4.0],
            [4.0, 0.0, 0.0],
        ]
    )
    captured: dict[str, np.ndarray] = {}

    class FakePCA:
        def __init__(self, n_components: int, random_state: int):
            self.n_components = n_components

        def fit_transform(self, values: np.ndarray) -> np.ndarray:
            captured["pca"] = values.copy()
            return values[:, : self.n_components]

    class FakeUMAP:
        def __init__(self, **kwargs):
            self.n_components = kwargs["n_components"]

        def fit_transform(self, values: np.ndarray) -> np.ndarray:
            captured["umap"] = values.copy()
            return np.zeros((values.shape[0], self.n_components), dtype=values.dtype)

    monkeypatch.setattr(analysis_module, "PCA", FakePCA)
    monkeypatch.setitem(sys.modules, "umap", types.SimpleNamespace(UMAP=FakeUMAP))

    project_pca(vectors, n_components=2)
    np.testing.assert_allclose(
        captured["pca"],
        prepare_cluster_samples(vectors, center=True, normalize=True).numpy(),
    )

    project_umap(vectors, n_components=2)
    np.testing.assert_allclose(
        captured["umap"],
        prepare_cluster_samples(vectors, center=True, normalize=True).numpy(),
    )

    project_pca(vectors, n_components=2, normalize=False)
    np.testing.assert_allclose(
        captured["pca"],
        prepare_cluster_samples(vectors, center=True, normalize=False).numpy(),
    )

    project_umap(vectors, n_components=2, normalize=False)
    np.testing.assert_allclose(
        captured["umap"],
        prepare_cluster_samples(vectors, center=True, normalize=False).numpy(),
    )


def test_pca_scree_preprocessing_normalizes_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vectors = torch.tensor(
        [
            [2.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 4.0],
            [4.0, 0.0, 0.0],
        ]
    )
    captured: dict[str, np.ndarray] = {}

    class FakePCA:
        explained_variance_ratio_ = np.asarray([1.0])

        def __init__(self, n_components: int, random_state: int):
            pass

        def fit(self, values: np.ndarray):
            captured["scree"] = values.copy()
            return self

    monkeypatch.setattr(analysis_module, "PCA", FakePCA)

    pca_explained_variance(vectors, n_components=1)
    np.testing.assert_allclose(
        captured["scree"],
        prepare_cluster_samples(vectors, center=True, normalize=True).numpy(),
    )

    pca_explained_variance(vectors, n_components=1, normalize=False)
    np.testing.assert_allclose(
        captured["scree"],
        prepare_cluster_samples(vectors, center=True, normalize=False).numpy(),
    )


def test_probe_regression_baseline_uses_held_out_split() -> None:
    X = np.arange(60, dtype=np.float32).reshape(10, 6)
    y = np.asarray([0.0, 1.0, 3.0, 6.0, 10.0, 15.0, 21.0, 28.0, 36.0, 45.0])

    metrics = evaluate_regression(X, y, layer=0, seed=0)
    _, test_idx = train_test_split(
        np.arange(len(y)),
        test_size=0.2,
        random_state=0,
        stratify=None,
    )
    y_test = y[test_idx]
    baseline = np.full_like(y_test, float(y_test.mean()))

    assert metrics["baseline_mae"] == pytest.approx(
        mean_absolute_error(y_test, baseline)
    )


def test_probe_heatmap_picks_lower_mae_before_flipping() -> None:
    rows = {
        "age": [
            {
                "attribute": "age",
                "layer": 0,
                "probe_kind": "worse",
                "mae": 4.0,
                "baseline_mae": 5.0,
            },
            {
                "attribute": "age",
                "layer": 0,
                "probe_kind": "better",
                "mae": 2.0,
                "baseline_mae": 5.0,
            },
        ]
    }

    fig = plot_attribute_layer_selectivity_heatmap(rows, metric="mae")

    assert fig.data[0].z[0][0] == pytest.approx(3.0)


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
    with pytest.raises(ValueError, match="projection_data kind"):
        build_layered_figure(
            samples,
            "umap",
            layers=[0, 1],
            projection_data=projection_data,
        )
    with pytest.raises(ValueError, match="projection_data normalize"):
        build_layered_figure(
            samples,
            "pca",
            layers=[0, 1],
            projection_normalize=False,
            projection_data=projection_data,
        )


def test_layered_projection_data_validates_graph_settings() -> None:
    samples = _layered_samples()
    projection_data = prepare_layered_projection_data(
        samples,
        "isomap",
        layers=[0, 1],
        graph_overlay=True,
        graph_n_neighbors=2,
    )

    with pytest.raises(ValueError, match="graph_n_neighbors"):
        build_layered_figure(
            samples,
            "isomap",
            layers=[0, 1],
            graph_overlay=True,
            graph_n_neighbors=3,
            projection_data=projection_data,
        )


def test_load_analysis_dataset_aligns_metadata_and_vectors() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = PersonaVectorStore("test/model", root_dir=tmp)
        for idx in range(2):
            store.save(
                "templated",
                f"persona-{idx:03d}",
                f"Persona {idx}",
                torch.randn(3, 6),
                ["q0"],
            )
        dataset = load_analysis_dataset(store, ["templated"])
        assert isinstance(dataset, AnalysisDataset)
        assert dataset.layers == (0, 1, 2)
        assert dataset.persona_names == {
            "persona-000": "Persona 0",
            "persona-001": "Persona 1",
        }
        assert dataset.samples("templated").vectors.shape == (2, 3, 6)


def test_probe_artifact_roundtrip_uses_canonical_schema() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        labels = AttributeLabels(
            attribute_name="sex",
            task="binary",
            y=np.asarray([0, 1, 0, 1]),
            labels=["f", "m", "f", "m"],
            class_names=["f", "m"],
        )
        directory = save_probe_artifact(
            X=np.asarray(
                [[0.0, 1.0], [1.0, 0.0], [0.1, 1.1], [1.1, 0.1]],
                dtype=np.float32,
            ),
            y=labels.y,
            labels=labels,
            task="binary",
            probe_kind="logistic_regression",
            layer=2,
            model_name="test/model",
            variant="templated",
            mask_strategy="answer_mean",
            output_dir=tmp,
        )
        artifact = load_probe_artifact(directory)
        assert artifact.schema_version == 2
        assert artifact.metadata["attribute_name"] == "sex"
        assert {
            "weight",
            "bias",
            "scaler_mean",
            "scaler_scale",
        } <= artifact.tensors.keys()


def test_pca_normalization_flag_controls_probe_pipeline() -> None:
    normalized = make_linear_probe(
        "logistic_regression",
        n_pca_components=2,
        normalize_pca=True,
    )
    raw_pca = make_linear_probe(
        "logistic_regression",
        n_pca_components=2,
        normalize_pca=False,
    )
    raw_full = make_linear_probe("logistic_regression", normalize_pca=False)

    assert list(normalized.named_steps) == ["scale", "pca", "probe"]
    assert list(raw_pca.named_steps) == ["pca", "probe"]
    assert list(raw_full.named_steps) == ["scale", "probe"]


def test_probe_artifact_pca_without_normalization_omits_scaler() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        labels = AttributeLabels(
            attribute_name="sex",
            task="binary",
            y=np.asarray([0, 1, 0, 1]),
            labels=["f", "m", "f", "m"],
            class_names=["f", "m"],
        )
        X = np.asarray(
            [
                [0.0, 1.0, 2.0],
                [1.0, 0.0, 3.0],
                [0.1, 1.1, 2.1],
                [1.1, 0.1, 3.1],
            ],
            dtype=np.float32,
        )
        directory = save_probe_artifact(
            X=X,
            y=labels.y,
            labels=labels,
            task="binary",
            probe_kind="logistic_regression",
            layer=2,
            model_name="test/model",
            variant="templated",
            mask_strategy="answer_mean",
            n_pca_components=2,
            normalize_pca=False,
            output_dir=tmp,
        )

        artifact = load_probe_artifact(directory)
        assert artifact.metadata["normalize_pca"] is False
        assert "scaler_mean" not in artifact.tensors
        assert "scaler_scale" not in artifact.tensors
        torch.testing.assert_close(
            artifact.tensors["pca_mean"],
            torch.from_numpy(X.mean(axis=0)),
        )


def test_prepare_kmeans_groups_can_be_passed_as_groups() -> None:
    samples = _layered_samples()
    groups = prepare_kmeans_groups(
        samples,
        layers=[0, 1],
        n_clusters=2,
        cluster_mode="per_layer",
    )

    fig = build_layered_figure(samples, "pca", layers=[0, 1], groups=groups)

    assert isinstance(groups, dict)
    assert sorted(groups) == [0, 1]
    assert [frame.name for frame in fig.frames] == ["0", "1"]


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
        store = PersonaVectorStore(
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

        hub_store = HFPersonaVectorStore("test/repo", "test/model")
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
            raise AssertionError("HFPersonaVectorStore should reject mismatched masks")

        store.save(
            "templated",
            BASELINE_PERSONA_ID,
            BASELINE_PERSONA_NAME,
            vectors,
            sample_ids,
        )
        assert store.list_personas(["templated"]) == ["persona-001"]
        assert store.list_personas(["templated"], include_baseline=True) == [
            BASELINE_PERSONA_ID,
            "persona-001",
        ]

    with tempfile.TemporaryDirectory() as tmp:
        store = PersonaVectorStore("test/model", root_dir=tmp)
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
        store = PersonaVectorStore("test/model", root_dir=tmp)
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


def test_hf_metadata_uses_vectorized_columnar_scan() -> None:
    class ColumnarDataset(list):
        def __init__(self, rows):
            super().__init__(rows)
            self.column_reads = 0

        def select_columns(self, columns):
            source = self

            class Selected:
                def __len__(self):
                    return len(source)

                def __getitem__(self, column: str):
                    # Reject per-row indexing: the scan must be columnar.
                    assert isinstance(column, str), "metadata scan must be columnar"
                    source.column_reads += 1
                    return [row[column] for row in source]

            return Selected()

    vector = torch.zeros(2, 3)
    dataset = ColumnarDataset(
        [
            {
                "persona_id": f"persona-{idx:03d}",
                "name": f"Persona {idx}",
                "vector": vector.tolist(),
            }
            for idx in range(20)
        ]
    )
    store = HFPersonaVectorStore("test/repo", "test/model")
    store._datasets["templated"] = dataset

    assert store.persona_names(
        ["persona-001", "persona-003"], variants=["templated"]
    ) == {
        "persona-001": "Persona 1",
        "persona-003": "Persona 3",
    }
    # One vectorized pass reads exactly the two metadata columns, once.
    assert dataset.column_reads == 2

    # Every subsequent metadata query is served from the cache.
    assert torch.allclose(store.load("templated", "persona-001"), vector)
    assert store.persona_names(["persona-007"], variants=["templated"]) == {
        "persona-007": "Persona 7",
    }
    assert len(store.list_personas(["templated"])) == 20
    assert store.list_personas(["templated"])[:2] == ["persona-000", "persona-001"]
    assert dataset.column_reads == 2
