from __future__ import annotations

import re
from dataclasses import dataclass


CANONICAL_VISIBILITY = {"good", "adequate", "marginal", "poor", "dark", "unknown"}
CANONICAL_ATTRIBUTION = {"certain", "mixed", "uncertain", "impossible_due_to_confounding"}
CANONICAL_PUBLIC_REGIONS = {
    "road",
    "footpath",
    "crossing",
    "curb",
    "median",
    "verge",
    "vegetation",
    "vehicle",
    "building_frontage",
    "shopfront",
    "window",
    "sign_billboard",
    "traffic_signal",
    "sky",
    "wet_reflection_like_road",
    "occluder",
    "unknown",
}
CANONICAL_AFFECTED_REGIONS = {
    "affected_road",
    "affected_footpath",
    "affected_crossing",
    "affected_verge",
    "lit_area",
    "unknown",
}
CANONICAL_LAMP_STATUS = {"on", "dim", "off", "flicker", "occluded", "saturated", "unknown"}


@dataclass(frozen=True)
class NormalizedValue:
    raw: str
    normalized: str
    warning: str | None = None

    @property
    def valid(self) -> bool:
        return self.warning is None


def _clean(value: object) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("-", "_").replace("/", "_")
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def normalize_visibility(value: object) -> NormalizedValue:
    raw = _clean(value)
    mapped = {"limited": "marginal", "partial": "marginal", "partly_visible": "marginal"}.get(raw, raw)
    if mapped in CANONICAL_VISIBILITY:
        return NormalizedValue(raw=raw, normalized=mapped)
    return NormalizedValue(raw=raw, normalized="unknown", warning=f"unsupported visibility label: {raw or '<empty>'}")


def normalize_attribution(value: object) -> NormalizedValue:
    raw = _clean(value)
    mapped = {
        "streetlight_primary": "certain",
        "target_lamp_primary": "certain",
        "unclear": "uncertain",
        "unknown": "uncertain",
        "likely_target": "certain",
        "likely_target_lamp": "certain",
        "likely_non_target_source": "impossible_due_to_confounding",
    }.get(raw, raw)
    if mapped in CANONICAL_ATTRIBUTION:
        return NormalizedValue(raw=raw, normalized=mapped)
    return NormalizedValue(raw=raw, normalized="uncertain", warning=f"unsupported attribution label: {raw or '<empty>'}")


def normalize_lamp_status(value: object) -> NormalizedValue:
    raw = _clean(value)
    if raw in CANONICAL_LAMP_STATUS:
        return NormalizedValue(raw=raw, normalized=raw)
    return NormalizedValue(raw=raw, normalized="unknown", warning=f"unsupported lamp status label: {raw or '<empty>'}")


def normalize_confounder(value: object) -> NormalizedValue:
    raw = _clean(value)
    if not raw:
        return NormalizedValue(raw=raw, normalized="unknown_bright_source", warning="empty confounder label")
    if any(token in raw for token in ("headlight", "tail_light", "taillight", "vehicle_light", "scooter_light", "motorcycle_light")):
        return NormalizedValue(raw=raw, normalized="headlight")
    if any(token in raw for token in ("shop", "window", "building", "facade", "wall", "canopy", "fuel_station")):
        return NormalizedValue(raw=raw, normalized="shopfront_or_window")
    if any(token in raw for token in ("sign", "banner", "signal", "billboard", "traffic_barrier", "barricade")):
        return NormalizedValue(raw=raw, normalized="sign_or_signal")
    if any(token in raw for token in ("wet", "reflection", "reflective", "glare", "flare", "glass")):
        if "lens_flare" in raw:
            return NormalizedValue(raw=raw, normalized="unknown_bright_source")
        return NormalizedValue(raw=raw, normalized="reflection")
    if "streetlight" in raw or "other_lamp" in raw:
        return NormalizedValue(raw=raw, normalized="other_lamp")
    if "unknown" in raw or "bright_source" in raw or "point_light" in raw:
        return NormalizedValue(raw=raw, normalized="unknown_bright_source")
    return NormalizedValue(raw=raw, normalized="unknown_bright_source", warning=f"unmapped confounder label: {raw}")


def normalize_public_region(value: object) -> NormalizedValue:
    raw = _clean(value)
    mapped = {
        "footpath_sidewalk": "footpath",
        "sidewalk": "footpath",
        "building": "building_frontage",
        "frontage": "building_frontage",
        "sign": "sign_billboard",
        "signage": "sign_billboard",
        "billboard": "sign_billboard",
        "wet_reflection": "wet_reflection_like_road",
        "wet_road_reflection": "wet_reflection_like_road",
    }.get(raw, raw)
    if mapped in CANONICAL_PUBLIC_REGIONS:
        return NormalizedValue(raw=raw, normalized=mapped)
    return NormalizedValue(raw=raw, normalized="unknown", warning=f"unsupported public-space region: {raw or '<empty>'}")


def normalize_affected_region(value: object) -> NormalizedValue:
    raw = _clean(value)
    mapped = {
        "road": "affected_road",
        "footpath": "affected_footpath",
        "sidewalk": "affected_footpath",
        "footpath_sidewalk": "affected_footpath",
        "crossing": "affected_crossing",
        "verge": "affected_verge",
        "public_edge": "affected_verge",
    }.get(raw, raw)
    if mapped in CANONICAL_AFFECTED_REGIONS:
        return NormalizedValue(raw=raw, normalized=mapped)
    return NormalizedValue(raw=raw, normalized="unknown", warning=f"unsupported affected-region type: {raw or '<empty>'}")

