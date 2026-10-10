from __future__ import annotations

from rbccps_od.config.model_registry import DEFAULT_MODEL_REGISTRY
from rbccps_od.config.schemas import ModelAssetSpec


def get_registry() -> dict[str, ModelAssetSpec]:
    return dict(DEFAULT_MODEL_REGISTRY)


def get_asset(name: str) -> ModelAssetSpec:
    registry = get_registry()
    if name not in registry:
        raise KeyError(f"Unknown model asset: {name}")
    return registry[name]
