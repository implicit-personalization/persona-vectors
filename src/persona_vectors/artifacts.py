import json
import os
import warnings
from pathlib import Path

import torch
from persona_data.prompts import BASELINE_PERSONA_ID, BASELINE_PERSONA_NAME
from safetensors.torch import load_file, save_file

PERSONA_VARIANTS: tuple[str, ...] = ("templated", "biography")
SUPPORTED_VARIANTS: tuple[str, ...] = (*PERSONA_VARIANTS, BASELINE_PERSONA_ID)
DEFAULT_MASK_STRATEGY = "answer_mean"
_MANIFEST_FILENAME = "manifest.json"
_TENSOR_KEY = "activations"
_TENSOR_SUFFIX = ".safetensors"


def model_dir_name(model_name: str) -> str:
    return model_name.replace("/", "__")


def _normalize_mask_strategy(mask_strategy: object | None) -> str:
    if mask_strategy is None:
        return DEFAULT_MASK_STRATEGY
    return str(getattr(mask_strategy, "value", mask_strategy))


def _variant_root(
    root_dir: str | Path,
    model_name: str,
    prompt_variant: str,
    mask_strategy: object | None,
) -> Path:
    return (
        Path(root_dir)
        / model_dir_name(model_name)
        / _normalize_mask_strategy(mask_strategy)
        / prompt_variant
    )


def _persona_tensor_path(variant_root: Path, persona_id: str) -> Path:
    if "/" in persona_id or "\\" in persona_id:
        raise ValueError(f"persona_id cannot contain path separators: {persona_id!r}")
    return variant_root / f"{persona_id}{_TENSOR_SUFFIX}"


def _load_manifest(variant_root: Path) -> dict:
    manifest_file = variant_root / _MANIFEST_FILENAME
    if not manifest_file.exists():
        raise FileNotFoundError(manifest_file)
    manifest = json.loads(manifest_file.read_text())
    personas = manifest.get("personas")
    if not isinstance(personas, dict):
        raise ValueError(f"manifest {manifest_file} is missing a personas mapping")
    return manifest


def _variant_manifests(
    root_dir: str | Path,
    model_name: str,
    variants: list[str],
    mask_strategy: object | None,
) -> list[dict]:
    variant_roots = [
        _variant_root(root_dir, model_name, variant, mask_strategy)
        for variant in variants
    ]
    if not variant_roots or any(
        not (root / _MANIFEST_FILENAME).exists() for root in variant_roots
    ):
        return []
    return [_load_manifest(root) for root in variant_roots]


class ActivationStore:
    """Artifact storage for masked-mean activation vectors."""

    def __init__(
        self,
        model_name: str,
        root_dir: str | Path | None = None,
        mask_strategy: object | None = DEFAULT_MASK_STRATEGY,
        variants: list[str] | tuple[str, ...] = PERSONA_VARIANTS,
    ) -> None:
        self.model_name = model_name
        self.mask_strategy = mask_strategy
        self.variants = tuple(variants)
        self.root_dir = (
            Path(root_dir)
            if root_dir is not None
            else Path(os.environ.get("ARTIFACTS_DIR", "artifacts")) / "activations"
        )

    def _mask_strategy(self, mask_strategy: object | None) -> object:
        return self.mask_strategy if mask_strategy is None else mask_strategy

    def save(
        self,
        prompt_variant: str,
        persona_id: str,
        persona_name: str,
        per_question_vectors: torch.Tensor,
        sample_ids: list[str],
        mask_strategy: object | None = None,
    ) -> Path:
        if per_question_vectors.ndim != 3:
            raise ValueError(
                "per_question_vectors must have shape (n_samples, num_layers, hidden_size)"
            )
        if len(sample_ids) != per_question_vectors.shape[0]:
            raise ValueError("number of sample ids must match first tensor dimension")
        if prompt_variant == BASELINE_PERSONA_ID:
            persona_id = BASELINE_PERSONA_ID
            persona_name = BASELINE_PERSONA_NAME

        variant_root = _variant_root(
            self.root_dir,
            self.model_name,
            prompt_variant,
            self._mask_strategy(mask_strategy),
        )
        variant_root.mkdir(parents=True, exist_ok=True)

        manifest_path = variant_root / _MANIFEST_FILENAME
        manifest = (
            _load_manifest(variant_root)
            if manifest_path.exists()
            else {
                "num_layers": int(per_question_vectors.shape[1]),
                "hidden_size": int(per_question_vectors.shape[2]),
                "personas": {},
            }
        )
        if manifest.get("num_layers") != int(
            per_question_vectors.shape[1]
        ) or manifest.get("hidden_size") != int(per_question_vectors.shape[2]):
            raise ValueError(
                f"tensor shape for {persona_id!r} does not match existing artifact manifest"
            )

        tensor_path = _persona_tensor_path(variant_root, persona_id)
        save_file({_TENSOR_KEY: per_question_vectors.detach().cpu()}, str(tensor_path))

        manifest["num_layers"] = int(per_question_vectors.shape[1])
        manifest["hidden_size"] = int(per_question_vectors.shape[2])
        if prompt_variant == BASELINE_PERSONA_ID:
            manifest["personas"] = {}
        manifest.setdefault("personas", {})[persona_id] = {
            "name": persona_name,
            "sample_ids": list(sample_ids),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))
        return variant_root

    def load(
        self,
        prompt_variant: str,
        persona_id: str,
        mask_strategy: object | None = None,
    ) -> tuple[torch.Tensor, list[str]]:
        requested = _normalize_mask_strategy(self._mask_strategy(mask_strategy))
        variant_root = _variant_root(
            self.root_dir, self.model_name, prompt_variant, requested
        )
        manifest = _load_manifest(variant_root)
        entry = manifest["personas"].get(persona_id)
        if not isinstance(entry, dict):
            raise FileNotFoundError(
                f"No activations found for {self.model_name!r} / {prompt_variant!r} / {requested!r} / {persona_id!r}"
            )

        tensor_path = _persona_tensor_path(variant_root, persona_id)
        if not tensor_path.exists():
            raise FileNotFoundError(tensor_path)
        tensors = load_file(str(tensor_path))
        if _TENSOR_KEY not in tensors:
            raise FileNotFoundError(f"Missing {_TENSOR_KEY!r} tensor in {tensor_path}")

        vectors = tensors[_TENSOR_KEY]
        if vectors.ndim != 3:
            raise ValueError(
                f"tensor for {persona_id!r} must have shape (n_samples, num_layers, hidden_size)"
            )

        sample_ids = entry.get("sample_ids")
        if not isinstance(sample_ids, list):
            raise ValueError(f"manifest entry for {persona_id} is missing sample ids")
        if len(sample_ids) != vectors.shape[0]:
            raise ValueError(
                f"sample ids for {persona_id!r} do not match tensor length"
            )
        return vectors, [str(sample_id) for sample_id in sample_ids]

    def list_personas(
        self,
        variants: list[str] | tuple[str, ...] | None = None,
        mask_strategy: object | None = None,
        warn_missing: bool = True,
    ) -> list[str]:
        """Return persona ids available in every requested variant."""

        requested_variants = self.variants if variants is None else variants
        return list_personas(
            self.root_dir,
            self.model_name,
            list(requested_variants),
            mask_strategy=self._mask_strategy(mask_strategy),
            warn_missing=warn_missing,
        )

    def persona_names(
        self,
        persona_ids: list[str],
        variants: list[str] | tuple[str, ...] | None = None,
        mask_strategy: object | None = None,
    ) -> dict[str, str]:
        """Return display names for known persona ids from saved manifests.

        Results follow ``persona_ids`` order, so ``list(names.values())`` is
        stable on supported Python versions.
        """

        requested_variants = self.variants if variants is None else variants
        return load_persona_names(
            self.root_dir,
            self.model_name,
            list(requested_variants),
            persona_ids,
            mask_strategy=self._mask_strategy(mask_strategy),
        )

    def available_variants(
        self,
        variants: list[str] | tuple[str, ...] | None = None,
        mask_strategy: object | None = None,
    ) -> list[str]:
        """Return candidate variants that have at least one saved persona."""

        candidate_variants = self.variants if variants is None else variants
        return [
            variant
            for variant in candidate_variants
            if self.list_personas(
                [variant], mask_strategy=mask_strategy, warn_missing=False
            )
        ]


def list_personas(
    root_dir: str | Path,
    model_name: str,
    variants: list[str],
    mask_strategy: object | None = DEFAULT_MASK_STRATEGY,
    warn_missing: bool = True,
) -> list[str]:
    """Return persona ids available in every requested variant.

    The result is the intersection across all ``variants`` for one
    model/mask-strategy artifact group. This keeps downstream comparisons from
    silently pairing a persona that exists in one prompt variant but not another.
    """
    if not variants:
        return []

    variant_roots = {
        variant: _variant_root(root_dir, model_name, variant, mask_strategy)
        for variant in variants
    }
    if any(not (root / _MANIFEST_FILENAME).exists() for root in variant_roots.values()):
        return []

    variant_personas = {
        variant: set(_load_manifest(root)["personas"].keys())
        for variant, root in variant_roots.items()
    }
    shared = set.intersection(*variant_personas.values())

    if warn_missing and len(variant_personas) > 1:
        all_personas = set.union(*variant_personas.values())
        skipped = len(all_personas - shared)
        if skipped:
            warnings.warn(
                f"Skipping {skipped} persona(s) missing one or more requested variants.",
                stacklevel=2,
            )

    return sorted(shared)


def load_persona_names(
    root_dir: str | Path,
    model_name: str,
    variants: list[str],
    persona_ids: list[str],
    mask_strategy: object | None = DEFAULT_MASK_STRATEGY,
) -> dict[str, str]:
    """Load display names for known persona ids from saved manifests.

    Names are looked up across ``variants`` in order; the first non-empty name
    wins. Missing manifests or missing persona entries are ignored.
    """
    manifests = _variant_manifests(root_dir, model_name, variants, mask_strategy)
    if not manifests:
        return {}

    names: dict[str, str] = {}
    for persona_id in persona_ids:
        for manifest in manifests:
            entry = manifest["personas"].get(persona_id)
            if isinstance(entry, dict):
                name = entry.get("name")
                if isinstance(name, str) and name:
                    names[persona_id] = name
                    break
    return names


def list_layers(
    root_dir: str | Path,
    model_name: str,
    variants: list[str],
    persona_ids: list[str],
    mask_strategy: object | None = DEFAULT_MASK_STRATEGY,
) -> list[int]:
    """Return shared layer indices for the requested artifacts.

    If ``persona_ids`` is provided, all ids must be present in every requested
    variant or an empty list is returned.
    """
    manifests = _variant_manifests(root_dir, model_name, variants, mask_strategy)
    if not manifests:
        return []

    if persona_ids:
        shared_personas = set.intersection(
            *(set(manifest["personas"].keys()) for manifest in manifests)
        )
        if not set(persona_ids) <= shared_personas:
            return []

    shared_layers: set[int] | None = None
    for manifest in manifests:
        num_layers = manifest.get("num_layers")
        if isinstance(num_layers, int) and num_layers >= 0:
            layers = set(range(num_layers))
            shared_layers = layers if shared_layers is None else shared_layers & layers
    return sorted(shared_layers or set())
