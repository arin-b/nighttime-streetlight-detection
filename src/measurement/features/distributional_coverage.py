from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rbccps_measurement.contracts.module_io import AffectedRegionFieldOutput, RISDecompositionOutput, SegmentationMaskOutput, SourceFieldOutput
from rbccps_measurement.decomposition.source_slots import SourceEvidence
from rbccps_measurement.geometry.lamp_footprint_field import FootprintEstimate
from rbccps_measurement.status.latent_emission_state import LampStatusEstimate


FEATURE_QUANTILES = ("q10", "q25", "q50", "q75", "q90")


@dataclass(frozen=True)
class DistributionalFeatureConfig:
    implementation: str = "deterministic_distributional_features_v1"
    base_useful_threshold: float = 0.34
    dark_hole_ratio: float = 0.70
    min_region_mass: float = 1e-6

    def to_dict(self) -> dict[str, Any]:
        return {
            "implementation": self.implementation,
            "base_useful_threshold": self.base_useful_threshold,
            "dark_hole_ratio": self.dark_hole_ratio,
            "min_region_mass": self.min_region_mass,
        }


@dataclass(frozen=True)
class UsefulIlluminationFeatures:
    coverage_proxy: float
    adequacy_proxy: float
    adequacy_class: str
    uniformity_proxy: float
    dark_hole_fraction: float
    glare_penalty: float
    confounder_penalty: float
    occlusion_penalty: float
    temporal_stability: float
    quantiles: dict[str, float] | None = None
    dark_hole_probability_map: np.ndarray | None = None
    utility_score: float = 0.0
    quality_by_region: dict[str, float] | None = None
    quality_flags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, float | str]:
        payload: dict[str, float | str] = {
            "coverage_proxy": round(self.coverage_proxy, 4),
            "adequacy_proxy": round(self.adequacy_proxy, 4),
            "adequacy_class": self.adequacy_class,
            "uniformity_proxy": round(self.uniformity_proxy, 4),
            "dark_hole_fraction": round(self.dark_hole_fraction, 4),
            "glare_penalty": round(self.glare_penalty, 4),
            "confounder_penalty": round(self.confounder_penalty, 4),
            "occlusion_penalty": round(self.occlusion_penalty, 4),
            "temporal_stability": round(self.temporal_stability, 4),
        }
        for key, value in (self.quantiles or {}).items():
            payload[f"illumination_{key}"] = round(float(value), 4)
        if self.quality_by_region:
            for key, value in self.quality_by_region.items():
                payload[f"region_quality_{key}"] = round(float(value), 4)
        payload["utility_score"] = round(float(self.utility_score), 4)
        return payload


def _clip01(value: float | np.ndarray) -> float | np.ndarray:
    return np.clip(value, 0.0, 1.0)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantiles: tuple[float, ...]) -> dict[str, float]:
    flat_values = values.reshape(-1).astype(np.float64)
    flat_weights = weights.reshape(-1).astype(np.float64)
    valid = flat_weights > 0
    if not np.any(valid):
        return {name: 0.0 for name in FEATURE_QUANTILES}
    flat_values = flat_values[valid]
    flat_weights = flat_weights[valid]
    order = np.argsort(flat_values)
    flat_values = flat_values[order]
    flat_weights = flat_weights[order]
    cumulative = np.cumsum(flat_weights)
    cumulative /= max(float(cumulative[-1]), 1e-12)
    result = np.interp(np.asarray(quantiles, dtype=np.float64), cumulative, flat_values)
    return {name: float(value) for name, value in zip(FEATURE_QUANTILES, result)}


def _region_threshold(region_mix: dict[str, float], base: float) -> float:
    thresholds = {
        "road": 0.36,
        "footpath": 0.28,
        "crossing": 0.42,
        "verge": 0.24,
    }
    if not region_mix:
        return base
    return float(sum(float(region_mix.get(key, 0.0)) * thresholds[key] for key in thresholds) or base)


def _fallback_features(
    status: LampStatusEstimate,
    footprint: FootprintEstimate,
    source: SourceEvidence,
    normalization_reliability: float,
) -> UsefulIlluminationFeatures:
    geometry_factor = {"good": 1.0, "moderate": 0.82, "weak": 0.55}[footprint.quality]
    coverage = max(0.0, min(1.0, status.confidence * geometry_factor * (1.0 - 0.4 * source.confounder_penalty)))
    uniformity = max(0.0, min(1.0, 0.75 * coverage + 0.25 * footprint.geometry_quality))
    glare = 0.28 if status.saturated_flag else 0.08
    occlusion = status.occluded_probability
    stability = max(0.0, min(1.0, 1.0 - status.flicker_index - 0.25 * (1.0 - normalization_reliability)))
    dark_hole = max(0.0, min(1.0, 1.0 - uniformity))
    adequacy = max(0.0, min(1.0, 0.45 * coverage + 0.30 * uniformity + 0.25 * stability - 0.2 * glare))
    return _package_features(coverage, adequacy, uniformity, dark_hole, glare, source.confounder_penalty, occlusion, stability, {}, None, {})


def estimate_useful_features(
    status: LampStatusEstimate,
    footprint: FootprintEstimate,
    source: SourceEvidence,
    normalization_reliability: float,
    segmentation: SegmentationMaskOutput | None = None,
    ris_output: RISDecompositionOutput | None = None,
    source_output: SourceFieldOutput | None = None,
    config: DistributionalFeatureConfig | None = None,
) -> UsefulIlluminationFeatures:
    config = config or DistributionalFeatureConfig()
    affected = footprint.field
    source_output = source_output or source.field_output
    if affected is None or ris_output is None:
        return _fallback_features(status, footprint, source, normalization_reliability)

    weights = np.clip(affected.affected_field.astype(np.float32), 0.0, 1.0)
    if float(np.sum(weights)) <= config.min_region_mass:
        flags = ("empty_affected_region",)
        fallback = _fallback_features(status, footprint, source, normalization_reliability)
        return UsefulIlluminationFeatures(**{**fallback.__dict__, "quality_flags": flags})

    illumination = np.clip(ris_output.illumination_like.astype(np.float32) + 0.35 * ris_output.source_like.astype(np.float32), 0.0, 1.0)
    threshold = _region_threshold(affected.region_mix, config.base_useful_threshold)
    quantiles = _weighted_quantile(illumination, weights, (0.10, 0.25, 0.50, 0.75, 0.90))
    coverage = float(np.sum(weights * (illumination >= threshold)) / max(float(np.sum(weights)), config.min_region_mass))
    dark_hole_map = np.clip(weights * (illumination < threshold * config.dark_hole_ratio), 0.0, 1.0).astype(np.float32)
    dark_hole = float(np.sum(dark_hole_map) / max(float(np.sum(weights)), config.min_region_mass))
    q10 = quantiles["q10"]
    q50 = max(quantiles["q50"], 1e-6)
    q90 = max(quantiles["q90"], 1e-6)
    low_percentile_balance = min(1.0, q10 / q50)
    spread_penalty = min(1.0, max(0.0, (q90 - q10) / max(q90, 1e-6)))
    uniformity = float(_clip01(0.68 * low_percentile_balance + 0.32 * (1.0 - spread_penalty)))

    source_output = source_output or source.field_output
    source_probs = source_output.source_probabilities if source_output is not None else {}
    glare = float(
        _clip01(
            (0.28 if status.saturated_flag else 0.06)
            + 0.22 * source_probs.get("reflection", 0.0)
            + 0.16 * source_probs.get("sign_or_signal", 0.0)
            + 0.10 * source_probs.get("unknown_bright_source", 0.0)
        )
    )
    occlusion = float(_clip01(0.55 * status.occluded_probability + 0.45 * np.mean(1.0 - affected.occlusion_gate)))
    stability = float(_clip01(1.0 - status.flicker_index - 0.25 * (1.0 - normalization_reliability)))
    confounder = float(_clip01(source.confounder_penalty))
    region_quality = _region_quality(illumination, affected, threshold)
    utility = float(_clip01(0.45 * coverage + 0.25 * uniformity + 0.20 * quantiles["q50"] + 0.10 * stability))
    adequacy = float(_clip01(utility - 0.18 * glare - 0.20 * confounder - 0.12 * occlusion))
    flags: list[str] = ["distributional_features"]
    if dark_hole > 0.35:
        flags.append("dark_hole_evidence")
    if confounder > 0.35:
        flags.append("confounder_limited_adequacy")
    if segmentation is not None and float(np.mean(segmentation.uncertainty_map)) > 0.45:
        flags.append("segmentation_uncertainty_affects_features")
    return _package_features(coverage, adequacy, uniformity, dark_hole, glare, confounder, occlusion, stability, quantiles, dark_hole_map, region_quality, utility, flags)


def _region_quality(illumination: np.ndarray, affected: AffectedRegionFieldOutput, threshold: float) -> dict[str, float]:
    regions = {
        "road": affected.road_region,
        "footpath": affected.footpath_region,
        "crossing": affected.crossing_region,
        "verge": affected.verge_region,
    }
    result: dict[str, float] = {}
    for name, mask in regions.items():
        weights = mask.astype(np.float32)
        total = float(np.sum(weights))
        if total <= 1e-9:
            result[name] = 0.0
        else:
            result[name] = float(np.sum(weights * (illumination >= threshold)) / total)
    return result


def _package_features(
    coverage: float,
    adequacy: float,
    uniformity: float,
    dark_hole: float,
    glare: float,
    confounder: float,
    occlusion: float,
    stability: float,
    quantiles: dict[str, float],
    dark_hole_map: np.ndarray | None,
    region_quality: dict[str, float],
    utility: float | None = None,
    flags: list[str] | None = None,
) -> UsefulIlluminationFeatures:
    if adequacy >= 0.72:
        adequacy_class = "adequate"
    elif adequacy >= 0.45:
        adequacy_class = "marginal"
    elif adequacy >= 0.2:
        adequacy_class = "poor"
    else:
        adequacy_class = "unknown"
    return UsefulIlluminationFeatures(
        coverage_proxy=float(_clip01(coverage)),
        adequacy_proxy=float(_clip01(adequacy)),
        adequacy_class=adequacy_class,
        uniformity_proxy=float(_clip01(uniformity)),
        dark_hole_fraction=float(_clip01(dark_hole)),
        glare_penalty=float(_clip01(glare)),
        confounder_penalty=float(_clip01(confounder)),
        occlusion_penalty=float(_clip01(occlusion)),
        temporal_stability=float(_clip01(stability)),
        quantiles=quantiles,
        dark_hole_probability_map=dark_hole_map,
        utility_score=float(_clip01(utility if utility is not None else adequacy)),
        quality_by_region=region_quality,
        quality_flags=tuple(sorted(set(flags or []))),
    )


class DistributionalFeatureEstimator:
    def __init__(self, checkpoint: dict[str, Any] | None = None) -> None:
        self.checkpoint = checkpoint or {"config": DistributionalFeatureConfig().to_dict()}
        self.config = _config_from_checkpoint(self.checkpoint)

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "DistributionalFeatureEstimator":
        checkpoint = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        return cls(checkpoint)

    def predict(self, *args: Any, **kwargs: Any) -> UsefulIlluminationFeatures:
        return estimate_useful_features(*args, config=self.config, **kwargs)


def _config_from_checkpoint(checkpoint: dict[str, Any]) -> DistributionalFeatureConfig:
    payload = checkpoint.get("config", {})
    return DistributionalFeatureConfig(
        implementation=str(payload.get("implementation", "deterministic_distributional_features_v1")),
        base_useful_threshold=float(payload.get("base_useful_threshold", 0.34)),
        dark_hole_ratio=float(payload.get("dark_hole_ratio", 0.70)),
        min_region_mass=float(payload.get("min_region_mass", 1e-6)),
    )
