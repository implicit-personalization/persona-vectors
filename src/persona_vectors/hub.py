"""Publish saved activation artifacts to the Hugging Face Hub."""

from pathlib import Path

from safetensors.torch import load_file

from persona_vectors.artifacts import (
    _MANIFEST_FILENAME,
    _TENSOR_KEY,
    DEFAULT_MASK_STRATEGY,
    SUPPORTED_VARIANTS,
    _load_manifest,
    _persona_tensor_path,
    _variant_root,
    activation_config_name,
)


def parse_vector_config_name(config_name: str) -> tuple[str, str] | None:
    """Parse a Hub vector dataset config into ``(model_name, mask_strategy)``."""

    model_key, separator, mask_strategy = config_name.rpartition("__")
    if not separator or not model_key or not mask_strategy:
        return None
    return model_key.replace("__", "/"), mask_strategy


def list_hub_vector_models(repo_id: str) -> dict[str, list[str]]:
    """Return available Hub vector models grouped by mask strategy."""
    from datasets import get_dataset_config_names

    models_by_strategy: dict[str, set[str]] = {}
    for config_name in get_dataset_config_names(repo_id):
        parsed = parse_vector_config_name(config_name)
        if parsed is None:
            continue
        model_name, mask_strategy = parsed
        models_by_strategy.setdefault(mask_strategy, set()).add(model_name)
    return {
        strategy: sorted(models)
        for strategy, models in sorted(models_by_strategy.items())
    }


def push_to_hub(
    repo_id: str,
    model_name: str,
    mask_strategy: object | None = DEFAULT_MASK_STRATEGY,
    root_dir: str | Path = "artifacts/activations",
    variants: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    """Push locally saved activations to a Hub dataset.

    One config per ``model__mask_strategy`` and one split per variant, matching
    the layout ``HFPersonaVectorStore`` reads. Returns the variants pushed.
    """
    from datasets import Array2D, Dataset, Features, Sequence, Value

    def build_split(variant_root: Path) -> Dataset:
        manifest = _load_manifest(variant_root)
        features = Features(
            {
                "persona_id": Value("string"),
                "name": Value("string"),
                "sample_ids": Sequence(Value("string")),
                "vector": Array2D(
                    shape=(int(manifest["num_layers"]), int(manifest["hidden_size"])),
                    dtype="float32",
                ),
            }
        )
        rows = [
            {
                "persona_id": persona_id,
                "name": entry["name"],
                "sample_ids": entry["sample_ids"],
                "vector": load_file(
                    str(_persona_tensor_path(variant_root, persona_id))
                )[_TENSOR_KEY]
                .float()
                .numpy(),
            }
            for persona_id, entry in sorted(manifest["personas"].items())
        ]
        return Dataset.from_list(rows, features=features)

    requested = list(variants) if variants else list(SUPPORTED_VARIANTS)
    config = activation_config_name(model_name, mask_strategy)
    pushed: list[str] = []
    for variant in requested:
        variant_root = _variant_root(root_dir, model_name, variant, mask_strategy)
        if not (variant_root / _MANIFEST_FILENAME).is_file():
            continue
        build_split(variant_root).push_to_hub(
            repo_id, config_name=config, split=variant
        )
        print(f"pushed {config}/{variant} -> https://huggingface.co/datasets/{repo_id}")
        pushed.append(variant)
    if not pushed:
        raise FileNotFoundError(
            f"no variant manifests for {model_name!r} under {root_dir}"
        )
    return pushed
