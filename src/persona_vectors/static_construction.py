from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from persona_vectors.artifacts import ActivationStore, list_personas
from persona_vectors.steering import _shared_item_key


def parse_layers(value: str, *, num_layers: int | None = None) -> list[int] | None:
    value = value.strip().lower()
    if value == "all":
        return None
    layers: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if end < start:
                raise ValueError(f"Invalid descending layer range: {part!r}")
            layers.extend(range(start, end + 1))
        else:
            layers.append(int(part))
    layers = sorted(dict.fromkeys(layers))
    if not layers:
        raise ValueError("--layers selected no layers")
    if num_layers is not None and (min(layers) < 0 or max(layers) >= num_layers):
        raise ValueError(f"Requested layers outside valid range [0, {num_layers - 1}]")
    return layers


def layers_label(layers: list[int] | None) -> str:
    return "all" if layers is None else "_".join(str(layer) for layer in layers)


def center_features(tensor: torch.Tensor) -> torch.Tensor:
    return tensor - tensor.mean(dim=-1, keepdim=True)


def l2_normalize_by_layer(tensor: torch.Tensor, eps: float) -> torch.Tensor:
    return tensor / (tensor.norm(dim=-1, keepdim=True) + eps)


def flatten(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.reshape(-1).float()


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(flatten(a), flatten(b), dim=0).item())


def resolve_activation_root(source_metadata: dict, *, cwd: Path | None = None) -> Path:
    activation_root = Path(source_metadata["activation_root"])
    if not activation_root.is_absolute():
        activation_root = (cwd or Path.cwd()) / activation_root
    return activation_root


def load_activation_records(
    *,
    store: ActivationStore,
    persona_ids: list[str],
    layers: list[int] | None,
    center: bool,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, str]]:
    records: dict[str, dict[str, torch.Tensor]] = {}
    names: dict[str, str] = {}
    for persona_id in persona_ids:
        activations, qids, questions = store.load_records("biography", persona_id)
        selected = activations.float() if layers is None else activations[:, layers, :].float()
        if center:
            selected = center_features(selected)
        metadata = store.load_metadata("biography", persona_id)
        names[persona_id] = metadata.get("persona_name", persona_id)

        item_map: dict[str, torch.Tensor] = {}
        for idx in range(selected.shape[0]):
            item_key = _shared_item_key(
                qid=qids[idx] if qids is not None else None,
                question=questions[idx],
                persona_id=persona_id,
            )
            item_map[item_key] = selected[idx]
        records[persona_id] = item_map
    return records, names


def build_item_banks(
    records: dict[str, dict[str, torch.Tensor]],
) -> dict[str, dict[str, torch.Tensor]]:
    item_banks: dict[str, dict[str, torch.Tensor]] = {}
    for persona_id, item_map in records.items():
        for item_key, activation in item_map.items():
            item_banks.setdefault(item_key, {})[persona_id] = activation
    return item_banks


def build_contrast_records(
    *,
    records: dict[str, dict[str, torch.Tensor]],
    item_banks: dict[str, dict[str, torch.Tensor]],
) -> dict[str, dict[str, torch.Tensor]]:
    contrasts: dict[str, dict[str, torch.Tensor]] = {}
    for persona_id, item_map in records.items():
        contrast_map: dict[str, torch.Tensor] = {}
        for item_key, positive in item_map.items():
            negatives = [
                activation
                for other_id, activation in item_banks[item_key].items()
                if other_id != persona_id
            ]
            if not negatives:
                continue
            contrast_map[item_key] = positive - torch.stack(negatives, dim=0).mean(dim=0)
        contrasts[persona_id] = contrast_map
    return contrasts


def compute_feature_variance(
    *,
    item_banks: dict[str, dict[str, torch.Tensor]],
    eps: float,
) -> torch.Tensor:
    residuals: list[torch.Tensor] = []
    for persona_acts in item_banks.values():
        if len(persona_acts) < 2:
            continue
        stack = torch.stack(list(persona_acts.values()), dim=0).float()
        residuals.extend((stack - stack.mean(dim=0, keepdim=True)).unbind(dim=0))
    if not residuals:
        raise ValueError("No residuals available for feature variance")
    residual_stack = torch.stack(residuals, dim=0)
    return residual_stack.var(dim=0, unbiased=False) + eps


def construct_vectors(
    *,
    contrasts: dict[str, dict[str, torch.Tensor]],
    feature_var: torch.Tensor,
    eps: float,
) -> dict[str, dict[str, torch.Tensor]]:
    vectors: dict[str, dict[str, torch.Tensor]] = {
        "raw_mean": {},
        "unit_mean": {},
        "diag_std_mean": {},
        "diag_var_mean": {},
    }
    feature_std = feature_var.sqrt()
    for persona_id, item_map in contrasts.items():
        stack = torch.stack(list(item_map.values()), dim=0).float()
        raw_mean = stack.mean(dim=0)
        unit_mean = l2_normalize_by_layer(stack, eps).mean(dim=0)
        unit_mean = unit_mean * (raw_mean.norm(dim=-1, keepdim=True) + eps)
        vectors["raw_mean"][persona_id] = raw_mean
        vectors["unit_mean"][persona_id] = unit_mean
        vectors["diag_std_mean"][persona_id] = (stack / feature_std).mean(dim=0)
        vectors["diag_var_mean"][persona_id] = (stack / feature_var).mean(dim=0)
    return vectors


def load_static_construction_inputs(
    *,
    model_name: str,
    activation_root: Path,
    layers: list[int] | None,
    center: bool,
) -> tuple[list[str], dict[str, dict[str, torch.Tensor]], dict[str, str]]:
    store = ActivationStore(model_name, activation_root)
    persona_ids = list_personas(activation_root, model_name, ["biography"])
    records, names = load_activation_records(
        store=store,
        persona_ids=persona_ids,
        layers=layers,
        center=center,
    )
    return persona_ids, records, names
