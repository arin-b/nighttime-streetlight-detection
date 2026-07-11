from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rbccps_measurement.contracts.module_io import AffectedRegionFieldOutput, SourceFieldOutput
from rbccps_measurement.decomposition.source_slots import SourceEvidence
from rbccps_measurement.features.distributional_coverage import UsefulIlluminationFeatures


@dataclass(frozen=True)
class CounterfactualAttributionConfig:
    implementation: str = "deterministic_counterfactual_attribution_v1"
    useful_threshold: float = 0.34
    mixed_competition_threshold: float = 0.35
    certain_score_threshold: float = 0.55

    def to_dict(self) -> dict[str, Any]:
        return {
            "implementation": self.implementation,
            "useful_threshold": self.useful_threshold,
            "mixed_competition_threshold": self.mixed_competition_threshold,
            "certain_score_threshold": self.certain_score_threshold,
        }


@dataclass(frozen=True)
class AttributionEstimate:
    score: float
    attribution_class: str
    uncertainty: float
    all_source_utility: float = 0.0
    without_target_utility: float = 0.0
    source_competition: dict[str, float] | None = None
    quality_flags: tuple[str, ...] = ()


def estimate_counterfactual_attribution(
    features: UsefulIlluminationFeatures,
    source: SourceEvidence,
    affected_region: AffectedRegionFieldOutput | None = None,
    source_output: SourceFieldOutput | None = None,
    config: CounterfactualAttributionConfig | None = None,
) -> AttributionEstimate:
    config = config or CounterfactualAttributionConfig()
    source_output = source_output or source.field_output
    if affected_region is None or source_output is None:
        return _fallback_attribution(features, source)

    weights = np.clip(affected_region.affected_field.astype(np.float32), 0.0, 1.0)
    total_weight = float(np.sum(weights))
    if total_weight <= 1e-9:
        fallback = _fallback_attribution(features, source)
        return AttributionEstimate(
            fallback.score,
            fallback.attribution_class,
            min(1.0, fallback.uncertainty + 0.25),
            fallback.all_source_utility,
            fallback.without_target_utility,
            fallback.source_competition,
            ("empty_affected_region_for_attribution",),
        )

    source_fields = source_output.source_fields
    target = source_fields.get("target_lamp", np.zeros_like(weights)).astype(np.float32)
    all_sources = np.sum(np.stack([field.astype(np.float32) for field in source_fields.values()], axis=0), axis=0)
    without_target = np.clip(all_sources - target, 0.0, 1.0)
    all_utility = _utility(weights, all_sources, config.useful_threshold)
    without_utility = _utility(weights, without_target, config.useful_threshold)
    alpha = max(0.0, all_utility - without_utility)
    normalized_alpha = alpha / max(all_utility, 1e-6)
    target_mass = float(np.sum(weights * target) / max(float(np.sum(weights * all_sources)), 1e-6))
    score = float(np.clip(0.62 * normalized_alpha + 0.38 * target_mass, 0.0, 1.0))
    competition = _source_competition(weights, source_fields)
    max_competitor = max((value for key, value in competition.items() if key != "target_lamp"), default=0.0)
    flags: list[str] = ["counterfactual_utility_difference"]
    if max_competitor >= config.mixed_competition_threshold:
        flags.append("source_competition_evidence")
    if score >= config.certain_score_threshold and max_competitor < config.mixed_competition_threshold and source.confounder_penalty < 0.35:
        klass = "certain"
    elif max_competitor >= config.mixed_competition_threshold or source.confounder_penalty >= 0.35:
        klass = "mixed"
    else:
        klass = "uncertain"
    uncertainty = float(np.clip(1.0 - score + 0.45 * max_competitor + 0.35 * source.confounder_penalty + source_output.source_confusion_score * 0.2, 0.0, 1.0))
    return AttributionEstimate(
        score=score,
        attribution_class=klass,
        uncertainty=uncertainty,
        all_source_utility=all_utility,
        without_target_utility=without_utility,
        source_competition=competition,
        quality_flags=tuple(sorted(set(flags))),
    )


def _utility(weights: np.ndarray, source_field: np.ndarray, threshold: float) -> float:
    useful = np.clip(source_field / max(threshold, 1e-6), 0.0, 1.0)
    return float(np.sum(weights * useful) / max(float(np.sum(weights)), 1e-9))


def _source_competition(weights: np.ndarray, fields: dict[str, np.ndarray]) -> dict[str, float]:
    masses = {name: float(np.sum(weights * field.astype(np.float32))) for name, field in fields.items()}
    total = sum(masses.values())
    if total <= 1e-9:
        return {name: 0.0 for name in fields}
    return {name: round(value / total, 6) for name, value in masses.items()}


def _fallback_attribution(features: UsefulIlluminationFeatures, source: SourceEvidence) -> AttributionEstimate:
    all_utility = features.utility_score or features.adequacy_proxy
    without_target = max(0.0, all_utility * (1.0 - source.target_lamp))
    score = max(0.0, min(1.0, all_utility - without_target + 0.25 * source.target_lamp))
    if score >= 0.65 and source.confounder_penalty < 0.25:
        klass = "certain"
    elif source.confounder_penalty >= 0.35:
        klass = "mixed"
    else:
        klass = "uncertain"
    uncertainty = max(0.0, min(1.0, 1.0 - score + 0.5 * source.confounder_penalty))
    return AttributionEstimate(score=score, attribution_class=klass, uncertainty=uncertainty, all_source_utility=all_utility, without_target_utility=without_target)


class CounterfactualAttributionEstimator:
    def __init__(self, checkpoint: dict[str, Any] | None = None) -> None:
        self.checkpoint = checkpoint or {"config": CounterfactualAttributionConfig().to_dict()}
        self.config = _config_from_checkpoint(self.checkpoint)

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "CounterfactualAttributionEstimator":
        checkpoint = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        return cls(checkpoint)

    def predict(self, *args: Any, **kwargs: Any) -> AttributionEstimate:
        return estimate_counterfactual_attribution(*args, config=self.config, **kwargs)


def _config_from_checkpoint(checkpoint: dict[str, Any]) -> CounterfactualAttributionConfig:
    payload = checkpoint.get("config", {})
    return CounterfactualAttributionConfig(
        implementation=str(payload.get("implementation", "deterministic_counterfactual_attribution_v1")),
        useful_threshold=float(payload.get("useful_threshold", 0.34)),
        mixed_competition_threshold=float(payload.get("mixed_competition_threshold", 0.35)),
        certain_score_threshold=float(payload.get("certain_score_threshold", 0.55)),
    )
