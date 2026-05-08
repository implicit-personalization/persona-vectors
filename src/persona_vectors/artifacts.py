import json
import os
import warnings
from collections.abc import Callable
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

PERSONA_VARIANTS: tuple[str, ...] = ("templated", "biography")
SUPPORTED_VARIANTS: tuple[str, ...] = PERSONA_VARIANTS
DEFAULT_MASK_STRATEGY = "answer_mean"
_MANIFEST_FILENAME = "manifest.json"
_TENSOR_KEY = "activations"
_TENSOR_SUFFIX = ".safetensors"


def model_dir_name(model_name: str) -> str:
    return model_name.replace("/", "__")


def normalize_mask_strategy(mask_strategy: object | None) -> str:
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
        / normalize_mask_strategy(mask_strategy)
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


def _shared_persona_ids(
    variant_personas: list[set[str]],
    warn_missing: bool = True,
) -> list[str]:
    """Return persona ids present in every variant persona set."""
    if not variant_personas:
        return []

    shared = set.intersection(*variant_personas)

    if warn_missing and len(variant_personas) > 1:
        all_personas = set.union(*variant_personas)
        skipped = len(all_personas - shared)
        if skipped:
            warnings.warn(
                f"Skipping {skipped} persona(s) missing one or more requested variants.",
                stacklevel=2,
            )

    return sorted(shared)


def _persona_names_from_variants(
    persona_ids: list[str],
    variants: list[str],
    name_lookup: Callable[[str, str], str | None],
) -> dict[str, str]:
    """Return first non-empty display name found while scanning variants."""
    names: dict[str, str] = {}
    for persona_id in persona_ids:
        for variant in variants:
            name = name_lookup(variant, persona_id)
            if isinstance(name, str) and name:
                names[persona_id] = name
                break
    return names


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
        vectors: torch.Tensor,
        sample_ids: list[str],
        mask_strategy: object | None = None,
    ) -> Path:
        """Save the mean activation vector for a persona.

        Args:
            prompt_variant: Prompt variant key (e.g. ``"templated"``).
            persona_id: Stable persona identifier.
            persona_name: Human-readable display name.
            vectors: Mean activation tensor of shape ``(num_layers, hidden_size)``,
                already averaged across questions and masked tokens.
            sample_ids: QA sample ids that were averaged into ``vectors`` — kept
                in the manifest for provenance, not used for indexing.
            mask_strategy: Mask strategy used during extraction. Defaults to the
                store's configured strategy.
        """
        if vectors.ndim != 2:
            raise ValueError("vectors must have shape (num_layers, hidden_size)")
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
                "num_layers": int(vectors.shape[0]),
                "hidden_size": int(vectors.shape[1]),
                "personas": {},
            }
        )
        if manifest.get("num_layers") != int(vectors.shape[0]) or manifest.get(
            "hidden_size"
        ) != int(vectors.shape[1]):
            raise ValueError(
                f"tensor shape for {persona_id!r} does not match existing artifact manifest"
            )

        tensor_path = _persona_tensor_path(variant_root, persona_id)
        save_file({_TENSOR_KEY: vectors.detach().cpu()}, str(tensor_path))

        manifest["num_layers"] = int(vectors.shape[0])
        manifest["hidden_size"] = int(vectors.shape[1])
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
    ) -> torch.Tensor:
        """Load the mean activation vector for a persona.

        Returns:
            Tensor of shape ``(num_layers, hidden_size)`` — the mean activation
            already averaged across questions and masked tokens at extraction time.

        Raises:
            FileNotFoundError: If no artifact exists for the requested combination.
        """
        requested = normalize_mask_strategy(self._mask_strategy(mask_strategy))
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
        if vectors.ndim != 2:
            raise ValueError(
                f"tensor for {persona_id!r} must have shape (num_layers, hidden_size)"
            )
        return vectors

    def list_personas(
        self,
        variants: list[str] | tuple[str, ...] | None = None,
        mask_strategy: object | None = None,
        warn_missing: bool = True,
    ) -> list[str]:
        """Return persona ids available in every requested variant."""

        requested_variants = variants or self.variants
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

        requested_variants = variants or self.variants
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

        candidate_variants = variants or self.variants
        return [
            variant
            for variant in candidate_variants
            if self.list_personas(
                [variant], mask_strategy=mask_strategy, warn_missing=False
            )
        ]


class HFActivationStore:
    """Read activation vectors from a Hugging Face dataset.

    The Hub dataset is expected to use one config per ``model__mask_strategy``
    and one split per prompt variant, matching ``persona_vectors.hub.push_to_hub``.
    """

    def __init__(
        self,
        repo_id: str,
        model_name: str,
        mask_strategy: object | None = DEFAULT_MASK_STRATEGY,
    ) -> None:
        self.repo_id = repo_id
        self.model_name = model_name
        self.mask_strategy = normalize_mask_strategy(mask_strategy)
        self.config_name = f"{model_dir_name(model_name)}__{self.mask_strategy}"
        self._cache: dict[str, dict[str, dict]] = {}

    def _variant(self, variant: str) -> dict[str, dict]:
        if variant not in self._cache:
            from datasets import load_dataset

            ds = load_dataset(self.repo_id, name=self.config_name, split=variant)
            self._cache[variant] = {row["persona_id"]: row for row in ds}
        return self._cache[variant]

    def _validate_mask_strategy(self, mask_strategy: object | None) -> None:
        if (
            mask_strategy is not None
            and normalize_mask_strategy(mask_strategy) != self.mask_strategy
        ):
            raise ValueError(
                f"HFActivationStore is bound to mask_strategy={self.mask_strategy!r}; "
                f"got {mask_strategy!r}"
            )

    def available_variants(
        self,
        variants: list[str] | tuple[str, ...] | None = None,
        mask_strategy: object | None = None,
    ) -> list[str]:
        """Return Hub splits available for this model and mask strategy."""
        from datasets import get_dataset_split_names

        self._validate_mask_strategy(mask_strategy)
        present = set(self._cache)
        try:
            present.update(
                get_dataset_split_names(self.repo_id, config_name=self.config_name)
            )
        except (FileNotFoundError, ValueError):
            pass
        candidates = list(variants) if variants else sorted(present)
        return [variant for variant in candidates if variant in present]

    def load(
        self,
        prompt_variant: str,
        persona_id: str,
        mask_strategy: object | None = None,
    ) -> torch.Tensor:
        """Load the mean activation vector for a persona from the Hub."""
        self._validate_mask_strategy(mask_strategy)
        rows = self._variant(prompt_variant)
        if persona_id not in rows:
            raise FileNotFoundError(
                f"{persona_id!r} not in {self.repo_id} {self.config_name}/{prompt_variant}"
            )
        return torch.as_tensor(rows[persona_id]["vector"], dtype=torch.float32)

    def list_personas(
        self,
        variants: list[str] | tuple[str, ...] | None = None,
        mask_strategy: object | None = None,
        warn_missing: bool = False,
    ) -> list[str]:
        """Return persona ids available in every requested Hub split."""
        self._validate_mask_strategy(mask_strategy)
        requested_variants = list(variants) if variants else self.available_variants()
        if not requested_variants:
            return []
        return _shared_persona_ids(
            [set(self._variant(variant).keys()) for variant in requested_variants],
            warn_missing=warn_missing,
        )

    def persona_names(
        self,
        persona_ids: list[str],
        variants: list[str] | tuple[str, ...] | None = None,
        mask_strategy: object | None = None,
    ) -> dict[str, str]:
        """Return display names for known persona ids from Hub rows."""
        self._validate_mask_strategy(mask_strategy)
        requested_variants = list(variants) if variants else self.available_variants()
        return _persona_names_from_variants(
            persona_ids,
            requested_variants,
            lambda variant, persona_id: (
                self._variant(variant).get(persona_id, {}).get("name")
            ),
        )


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

    return _shared_persona_ids(
        [
            set(_load_manifest(root)["personas"].keys())
            for root in variant_roots.values()
        ],
        warn_missing=warn_missing,
    )


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

    manifests_by_variant = dict(zip(variants, manifests, strict=False))

    def lookup(variant: str, persona_id: str) -> str | None:
        manifest = manifests_by_variant[variant]
        entry = manifest["personas"].get(persona_id)
        return entry.get("name") if isinstance(entry, dict) else None

    return _persona_names_from_variants(persona_ids, variants, lookup)


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
