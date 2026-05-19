import json
import warnings
from collections.abc import Callable
from pathlib import Path

import torch
from persona_data.environment import get_artifacts_dir
from persona_data.synth_persona import BASELINE_PERSONA_ID
from safetensors.torch import load_file, save_file

SUPPORTED_VARIANTS: tuple[str, ...] = ("templated", "biography")
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


def activation_config_name(
    model_name: str,
    mask_strategy: object | None = DEFAULT_MASK_STRATEGY,
) -> str:
    return f"{model_dir_name(model_name)}__{normalize_mask_strategy(mask_strategy)}"


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
    if not isinstance(manifest.get("personas"), dict):
        raise ValueError(f"manifest {manifest_file} is missing a personas mapping")
    return manifest


def _shared_persona_ids(
    variant_personas: list[set[str]],
    warn_missing: bool = True,
) -> list[str]:
    """Return persona ids present in every variant persona set."""
    if not variant_personas:
        return []
    shared = set.intersection(*variant_personas)
    if warn_missing and len(variant_personas) > 1:
        skipped = len(set.union(*variant_personas) - shared)
        if skipped:
            warnings.warn(
                f"Skipping {skipped} persona(s) missing one or more requested variants.",
                stacklevel=2,
            )
    return sorted(shared)


def _first_nonempty_name(
    persona_ids: list[str],
    variants: list[str],
    lookup: Callable[[str, str], str | None],
) -> dict[str, str]:
    """For each id, return the first non-empty name found across ``variants``."""
    names: dict[str, str] = {}
    for persona_id in persona_ids:
        for variant in variants:
            name = lookup(variant, persona_id)
            if isinstance(name, str) and name:
                names[persona_id] = name
                break
    return names


class PersonaVectorStore:
    """Local artifact storage for masked-mean persona vectors."""

    def __init__(
        self,
        model_name: str,
        root_dir: str | Path | None = None,
        mask_strategy: object | None = DEFAULT_MASK_STRATEGY,
        variants: list[str] | tuple[str, ...] = SUPPORTED_VARIANTS,
    ) -> None:
        self.model_name = model_name
        self.mask_strategy = mask_strategy
        self.variants = tuple(variants)
        self.root_dir = (
            Path(root_dir)
            if root_dir is not None
            else get_artifacts_dir() / "activations"
        )

    def _resolved_mask(self, mask_strategy: object | None) -> object:
        return self.mask_strategy if mask_strategy is None else mask_strategy

    def _root(self, prompt_variant: str, mask_strategy: object | None = None) -> Path:
        return _variant_root(
            self.root_dir,
            self.model_name,
            prompt_variant,
            self._resolved_mask(mask_strategy),
        )

    def _manifests(
        self,
        variants: list[str],
        mask_strategy: object | None = None,
    ) -> list[dict]:
        """Load manifests for ``variants``; returns ``[]`` if any are missing."""
        roots = [self._root(variant, mask_strategy) for variant in variants]
        if not roots or any(not (r / _MANIFEST_FILENAME).exists() for r in roots):
            return []
        return [_load_manifest(r) for r in roots]

    def save(
        self,
        prompt_variant: str,
        persona_id: str,
        persona_name: str,
        vectors: torch.Tensor,
        sample_ids: list[str],
        mask_strategy: object | None = None,
    ) -> Path:
        """Save a ``(num_layers, hidden_size)`` mean activation tensor. ``sample_ids`` are stored in the manifest for provenance."""
        if vectors.ndim != 2:
            raise ValueError("vectors must have shape (num_layers, hidden_size)")
        variant_root = self._root(prompt_variant, mask_strategy)
        variant_root.mkdir(parents=True, exist_ok=True)

        manifest_path = variant_root / _MANIFEST_FILENAME
        num_layers, hidden_size = int(vectors.shape[0]), int(vectors.shape[1])
        manifest = (
            _load_manifest(variant_root)
            if manifest_path.exists()
            else {"num_layers": num_layers, "hidden_size": hidden_size, "personas": {}}
        )
        if (
            manifest.get("num_layers") != num_layers
            or manifest.get("hidden_size") != hidden_size
        ):
            raise ValueError(
                f"tensor shape for {persona_id!r} does not match existing artifact manifest"
            )

        save_file(
            {_TENSOR_KEY: vectors.detach().cpu()},
            str(_persona_tensor_path(variant_root, persona_id)),
        )

        manifest["num_layers"] = num_layers
        manifest["hidden_size"] = hidden_size
        manifest["personas"][persona_id] = {
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
        """Load the saved ``(num_layers, hidden_size)`` mean activation tensor.

        Raises ``FileNotFoundError`` if no artifact exists for the combination.
        """
        variant_root = self._root(prompt_variant, mask_strategy)
        manifest = _load_manifest(variant_root)
        if not isinstance(manifest["personas"].get(persona_id), dict):
            requested = normalize_mask_strategy(self._resolved_mask(mask_strategy))
            raise FileNotFoundError(
                f"No activations found for {self.model_name!r} / {prompt_variant!r}"
                f" / {requested!r} / {persona_id!r}"
            )

        tensor_path = _persona_tensor_path(variant_root, persona_id)
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
        *,
        include_baseline: bool = False,
    ) -> list[str]:
        """Return persona ids available in every requested variant.

        Excludes ``baseline_assistant`` by default; pass
        ``include_baseline=True`` to keep it.
        """
        requested = self.variants if variants is None else tuple(variants)
        manifests = self._manifests(list(requested), mask_strategy)
        if not manifests:
            return []
        persona_ids = _shared_persona_ids(
            [set(m["personas"].keys()) for m in manifests],
            warn_missing=warn_missing,
        )
        if include_baseline:
            return persona_ids
        return [
            persona_id
            for persona_id in persona_ids
            if persona_id != BASELINE_PERSONA_ID
        ]

    def persona_sample_ids(
        self,
        prompt_variant: str,
        persona_id: str,
        mask_strategy: object | None = None,
    ) -> list[str] | None:
        """Return stored sample ids for a persona, if present."""
        variant_root = self._root(prompt_variant, mask_strategy)
        manifest_file = variant_root / _MANIFEST_FILENAME
        if not manifest_file.exists():
            return None
        manifest = _load_manifest(variant_root)
        entry = manifest["personas"].get(persona_id)
        if not isinstance(entry, dict):
            return None
        sample_ids = entry.get("sample_ids")
        if not isinstance(sample_ids, list):
            return None
        return [str(sample_id) for sample_id in sample_ids]

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
        requested = list(self.variants if variants is None else variants)
        manifests = self._manifests(requested, mask_strategy)
        if not manifests:
            return {}
        by_variant = dict(zip(requested, manifests, strict=True))

        def lookup(variant: str, persona_id: str) -> str | None:
            entry = by_variant[variant]["personas"].get(persona_id)
            return entry.get("name") if isinstance(entry, dict) else None

        return _first_nonempty_name(persona_ids, requested, lookup)

    def list_layers(
        self,
        variants: list[str] | tuple[str, ...],
        persona_ids: list[str] | tuple[str, ...] | None = None,
        mask_strategy: object | None = None,
    ) -> list[int]:
        """Return shared layer indices for the requested local artifacts."""
        manifests = self._manifests(list(variants), mask_strategy)
        if not manifests:
            return []

        if persona_ids:
            shared = set.intersection(*(set(m["personas"].keys()) for m in manifests))
            if not set(persona_ids) <= shared:
                return []

        shared_layers: set[int] | None = None
        for manifest in manifests:
            num_layers = manifest.get("num_layers")
            if isinstance(num_layers, int) and num_layers >= 0:
                layers = set(range(num_layers))
                shared_layers = (
                    layers if shared_layers is None else shared_layers & layers
                )
        return sorted(shared_layers or set())

    def available_variants(
        self,
        variants: list[str] | tuple[str, ...] | None = None,
        mask_strategy: object | None = None,
    ) -> list[str]:
        """Return candidate variants that have at least one saved persona."""
        candidates = self.variants if variants is None else variants
        return [
            variant
            for variant in candidates
            if self.list_personas(
                [variant],
                mask_strategy=mask_strategy,
                warn_missing=False,
                include_baseline=True,
            )
        ]


class HFPersonaVectorStore:
    """Read persona vectors from a Hugging Face dataset (lazy).

    The Hub dataset is expected to use one config per ``model__mask_strategy``
    and one split per prompt variant, matching ``persona_vectors.hub.push_to_hub``.

    Metadata queries (persona ids, names, layer count) load only the columns
    they need; the full vector array is fetched lazily on ``load()``.
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
        self.config_name = activation_config_name(model_name, self.mask_strategy)
        self._datasets: dict[str, object] = {}
        self._index: dict[str, dict[str, int]] = {}
        self._names: dict[str, dict[str, str]] = {}
        self._metadata_complete: set[str] = set()

    def _dataset(self, variant: str):
        if variant not in self._datasets:
            from datasets import load_dataset

            self._datasets[variant] = load_dataset(
                self.repo_id, name=self.config_name, split=variant
            )
        return self._datasets[variant]

    def _metadata(
        self, variant: str
    ) -> tuple[dict[str, int], dict[str, str]]:
        """Return ``(persona_id -> row_index, persona_id -> name)`` for ``variant``.

        Reads the ``persona_id``/``name`` columns in a single vectorized pass
        (the heavy ``vector`` column is excluded) and caches the result.
        """
        idx = self._index.setdefault(variant, {})
        names = self._names.setdefault(variant, {})
        if variant in self._metadata_complete:
            return idx, names

        meta = self._dataset(variant).select_columns(["persona_id", "name"])
        pids = meta["persona_id"]
        raw_names = meta["name"]
        for i, pid in enumerate(pids):
            idx[pid] = i
            names[pid] = raw_names[i] or pid
        self._metadata_complete.add(variant)
        return idx, names

    def _validate_mask_strategy(self, mask_strategy: object | None) -> None:
        if (
            mask_strategy is not None
            and normalize_mask_strategy(mask_strategy) != self.mask_strategy
        ):
            raise ValueError(
                f"HFPersonaVectorStore is bound to mask_strategy={self.mask_strategy!r};"
                f" got {mask_strategy!r}"
            )

    def release_cache(
        self, variants: list[str] | tuple[str, ...] | None = None
    ) -> None:
        """Drop cached datasets and metadata for ``variants`` (or all)."""
        targets = list(self._datasets) if variants is None else list(variants)
        for variant in targets:
            self._datasets.pop(variant, None)
            self._index.pop(variant, None)
            self._names.pop(variant, None)
            self._metadata_complete.discard(variant)

    def available_variants(
        self,
        variants: list[str] | tuple[str, ...] | None = None,
        mask_strategy: object | None = None,
    ) -> list[str]:
        """Return Hub splits available for this model and mask strategy."""
        from datasets import get_dataset_config_names, get_dataset_split_names

        self._validate_mask_strategy(mask_strategy)
        if variants is not None and not variants:
            return []
        present = set(self._datasets)
        try:
            present.update(
                get_dataset_split_names(self.repo_id, config_name=self.config_name)
            )
        except FileNotFoundError:
            pass
        except ValueError as exc:
            # Surface "config not found" instead of silently returning [].
            try:
                available = get_dataset_config_names(self.repo_id)
            except Exception:
                raise exc from None
            if self.config_name not in available:
                raise ValueError(
                    f"Config {self.config_name!r} not declared on {self.repo_id!r}. "
                    f"Available configs: {available}. "
                    "If the parquet files exist but no config is declared, the "
                    "dataset README likely lacks a `configs:` YAML block — "
                    "re-run scripts/upload_hf_readme.py to regenerate it."
                ) from exc
            raise
        candidates = sorted(present) if variants is None else list(variants)
        return [variant for variant in candidates if variant in present]

    def load(
        self,
        prompt_variant: str,
        persona_id: str,
        mask_strategy: object | None = None,
    ) -> torch.Tensor:
        """Load the mean activation vector for a persona from the Hub."""
        self._validate_mask_strategy(mask_strategy)
        idx, _ = self._metadata(prompt_variant)
        if persona_id not in idx:
            raise FileNotFoundError(
                f"{persona_id!r} not in {self.repo_id} {self.config_name}/{prompt_variant}"
            )
        row = self._dataset(prompt_variant)[idx[persona_id]]
        return torch.as_tensor(row["vector"], dtype=torch.float32)

    def list_personas(
        self,
        variants: list[str] | tuple[str, ...] | None = None,
        mask_strategy: object | None = None,
        warn_missing: bool = False,
        *,
        include_baseline: bool = False,
    ) -> list[str]:
        """Return persona ids available in every requested Hub split.

        Excludes ``baseline_assistant`` by default; pass
        ``include_baseline=True`` to keep it.
        """
        self._validate_mask_strategy(mask_strategy)
        requested = self.available_variants() if variants is None else list(variants)
        if not requested:
            return []
        persona_ids = _shared_persona_ids(
            [set(self._metadata(variant)[0]) for variant in requested],
            warn_missing=warn_missing,
        )
        if include_baseline:
            return persona_ids
        return [
            persona_id
            for persona_id in persona_ids
            if persona_id != BASELINE_PERSONA_ID
        ]

    def persona_names(
        self,
        persona_ids: list[str],
        variants: list[str] | tuple[str, ...] | None = None,
        mask_strategy: object | None = None,
    ) -> dict[str, str]:
        """Return display names for known persona ids from Hub rows."""
        self._validate_mask_strategy(mask_strategy)
        requested = self.available_variants() if variants is None else list(variants)
        name_maps = {
            variant: self._metadata(variant)[1] for variant in requested
        }
        return _first_nonempty_name(
            persona_ids,
            requested,
            lambda variant, pid: name_maps[variant].get(pid),
        )

    def list_layers(
        self,
        variants: list[str] | tuple[str, ...],
        persona_ids: list[str] | tuple[str, ...] | None = None,
        mask_strategy: object | None = None,
    ) -> list[int]:
        """Return shared layer indices for the requested Hub artifacts.

        Vectors within a variant are uniform shape, so the layer count is read
        from one sample persona per variant rather than scanning every row.
        """
        self._validate_mask_strategy(mask_strategy)
        if not variants:
            return []

        shared_layers: set[int] | None = None
        for variant in variants:
            idx, _ = self._metadata(variant)
            ids = list(persona_ids or []) or list(idx)
            if not ids or any(pid not in idx for pid in ids):
                return []
            sample_id = ids[0]
            vector = torch.as_tensor(self._dataset(variant)[idx[sample_id]]["vector"])
            if vector.ndim != 2:
                raise ValueError(
                    f"tensor for {sample_id!r} must have shape (num_layers, hidden_size)"
                )
            layers = set(range(int(vector.shape[0])))
            shared_layers = layers if shared_layers is None else shared_layers & layers
        return sorted(shared_layers or set())


def discover_activation_models(
    root_dir: str | Path,
    mask_strategy: object | None = DEFAULT_MASK_STRATEGY,
) -> list[str]:
    """Return model ids with at least one local artifact for ``mask_strategy``."""
    root = Path(root_dir).expanduser()
    if not root.is_dir():
        return []

    strategy = normalize_mask_strategy(mask_strategy)
    try:
        model_roots = sorted(p for p in root.iterdir() if p.is_dir())
    except OSError:
        return []

    models: list[str] = []
    for model_root in model_roots:
        strategy_root = model_root / strategy
        if not strategy_root.is_dir():
            continue
        try:
            has_manifest = any(
                (vr / _MANIFEST_FILENAME).is_file()
                for vr in strategy_root.iterdir()
                if vr.is_dir()
            )
        except OSError:
            continue
        if has_manifest:
            models.append(model_root.name.replace("__", "/"))
    return models


PersonaVectorSource = PersonaVectorStore | HFPersonaVectorStore
