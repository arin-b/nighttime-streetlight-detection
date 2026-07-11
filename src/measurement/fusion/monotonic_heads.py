from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rbccps_measurement.attribution.counterfactual import AttributionEstimate
from rbccps_measurement.contracts.module_io import SceneGraphArtifact, SceneGraphFusionOutput
from rbccps_measurement.features.distributional_coverage import UsefulIlluminationFeatures


ORDINAL_CLASSES = ("unknown", "poor", "marginal", "adequate")
EDGE_TYPES = ("serves", "overlaps", "confounds", "observed_by", "adjacent_to")


@dataclass(frozen=True)
class FusionResult:
    overall_score: float
    overall_category: str
    confidence: float
    fusion_output: SceneGraphFusionOutput | None = None


@dataclass(frozen=True)
class MonotonicFusionConfig:
    implementation: str = "deterministic_monotonic_scene_graph_fusion_v1"
    coverage_weight: float = 0.22
    adequacy_weight: float = 0.18
    uniformity_weight: float = 0.16
    stability_weight: float = 0.14
    attribution_weight: float = 0.20
    glare_penalty_weight: float = 0.13
    confounder_penalty_weight: float = 0.18
    occlusion_penalty_weight: float = 0.13
    dark_hole_penalty_weight: float = 0.10
    attribution_uncertainty_weight: float = 0.14
    missingness_penalty_weight: float = 0.10

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _category(score: float) -> str:
    if score >= 0.72:
        return "adequate"
    if score >= 0.45:
        return "marginal"
    if score >= 0.2:
        return "poor"
    return "unknown"


def build_scene_graph(
    track_id: str,
    features: UsefulIlluminationFeatures,
    attribution: AttributionEstimate,
    observation_completeness: float,
    region_mix: dict[str, float] | None = None,
    context: dict[str, Any] | None = None,
) -> SceneGraphArtifact:
    region_mix = region_mix or {}
    context = context or {}
    nodes: list[dict[str, Any]] = [
        {"id": f"lamp:{track_id}", "type": "lamp", "features": {"attribution": attribution.score, "status": context.get("status_score", 0.0)}},
        {
            "id": "region:served",
            "type": "region",
            "features": {
                "coverage": features.coverage_proxy,
                "adequacy": features.adequacy_proxy,
                "uniformity": features.uniformity_proxy,
                "dark_hole": features.dark_hole_fraction,
                **{f"mix_{key}": float(value) for key, value in region_mix.items()},
            },
        },
        {
            "id": "confounder:aggregate",
            "type": "confounder",
            "features": {
                "glare": features.glare_penalty,
                "confounder_penalty": features.confounder_penalty,
                "occlusion": features.occlusion_penalty,
            },
        },
        {
            "id": "camera:capture",
            "type": "camera",
            "features": {
                "metadata_quality": float(context.get("metadata_quality_score", 0.5)),
                "auto_exposure": float(context.get("auto_exposure", 0.0)),
            },
        },
        {
            "id": "geometry:context",
            "type": "geometry",
            "features": {
                "observation_completeness": observation_completeness,
                "geometry_quality": float(context.get("geometry_quality", 0.0)),
            },
        },
        {
            "id": "context:route",
            "type": "context",
            "features": {
                "route_known": float(bool(context.get("route_group"))),
                "confounder_density": features.confounder_penalty,
            },
        },
    ]
    edges = [
        {"source": f"lamp:{track_id}", "target": "region:served", "type": "serves", "weight": attribution.score},
        {"source": "region:served", "target": "confounder:aggregate", "type": "overlaps", "weight": features.confounder_penalty},
        {"source": "confounder:aggregate", "target": "region:served", "type": "confounds", "weight": features.confounder_penalty},
        {"source": "camera:capture", "target": f"lamp:{track_id}", "type": "observed_by", "weight": float(context.get("metadata_quality_score", 0.5))},
        {"source": "geometry:context", "target": "region:served", "type": "adjacent_to", "weight": float(context.get("geometry_quality", 0.0))},
    ]
    return SceneGraphArtifact(
        track_id=track_id,
        nodes=tuple(nodes),
        edges=tuple(edges),
        graph_features={
            "node_count": float(len(nodes)),
            "edge_count": float(len(edges)),
            "observation_completeness": float(observation_completeness),
            "confounder_density": float(features.confounder_penalty),
        },
        provenance={"implementation": "deterministic_scene_graph_artifact_v1"},
    )


def monotonic_fuse(
    features: UsefulIlluminationFeatures,
    attribution: AttributionEstimate,
    observation_completeness: float,
    track_id: str = "unknown_track",
    region_mix: dict[str, float] | None = None,
    context: dict[str, Any] | None = None,
    config: MonotonicFusionConfig | None = None,
) -> FusionResult:
    config = config or MonotonicFusionConfig()
    context = context or {}
    graph = build_scene_graph(track_id, features, attribution, observation_completeness, region_mix, context)
    missingness = _clip01(1.0 - observation_completeness + (1.0 - float(context.get("metadata_quality_score", 0.5))) * 0.35)
    component_scores = {
        "coverage": _clip01(features.coverage_proxy),
        "adequacy": _clip01(features.adequacy_proxy),
        "uniformity": _clip01(features.uniformity_proxy),
        "temporal_stability": _clip01(features.temporal_stability),
        "attribution": _clip01(attribution.score),
        "glare": _clip01(features.glare_penalty),
        "confounder": _clip01(features.confounder_penalty),
        "occlusion": _clip01(features.occlusion_penalty),
        "dark_hole": _clip01(features.dark_hole_fraction),
        "attribution_uncertainty": _clip01(attribution.uncertainty),
        "missingness": missingness,
    }
    contributions = {
        "coverage": config.coverage_weight * component_scores["coverage"],
        "adequacy": config.adequacy_weight * component_scores["adequacy"],
        "uniformity": config.uniformity_weight * component_scores["uniformity"],
        "temporal_stability": config.stability_weight * component_scores["temporal_stability"],
        "attribution": config.attribution_weight * component_scores["attribution"],
        "glare": -config.glare_penalty_weight * component_scores["glare"],
        "confounder": -config.confounder_penalty_weight * component_scores["confounder"],
        "occlusion": -config.occlusion_penalty_weight * component_scores["occlusion"],
        "dark_hole": -config.dark_hole_penalty_weight * component_scores["dark_hole"],
        "attribution_uncertainty": -config.attribution_uncertainty_weight * component_scores["attribution_uncertainty"],
        "missingness": -config.missingness_penalty_weight * component_scores["missingness"],
    }
    score = _clip01(sum(contributions.values()))
    uncertainty_index = _clip01(
        0.30 * component_scores["attribution_uncertainty"]
        + 0.20 * component_scores["confounder"]
        + 0.15 * component_scores["occlusion"]
        + 0.15 * component_scores["dark_hole"]
        + 0.20 * component_scores["missingness"]
    )
    raw_confidence = _clip01(observation_completeness * (1.0 - uncertainty_index) * (0.65 + 0.35 * float(context.get("metadata_quality_score", 0.5))))
    flags: list[str] = ["monotonic_scene_graph_fusion"]
    if uncertainty_index > 0.55:
        flags.append("high_fusion_uncertainty")
    if component_scores["confounder"] > 0.45:
        flags.append("fusion_confounder_limited")
    output = SceneGraphFusionOutput(
        track_id=track_id,
        graph=graph,
        component_scores=component_scores,
        monotonic_contributions={key: round(float(value), 6) for key, value in contributions.items()},
        raw_score=score,
        ordinal_class=_category(score),
        raw_confidence=raw_confidence,
        uncertainty_index=uncertainty_index,
        quality_flags=tuple(sorted(set(flags))),
        metadata={
            "implementation": config.implementation,
            "monotonic_constraints": {
                "score_positive": ["coverage", "adequacy", "uniformity", "temporal_stability", "attribution"],
                "score_negative": ["glare", "confounder", "occlusion", "dark_hole", "attribution_uncertainty", "missingness"],
                "confidence_negative": ["uncertainty_index"],
            },
            "ordinal_classes": list(ORDINAL_CLASSES),
        },
    )
    return FusionResult(score, output.ordinal_class, raw_confidence, fusion_output=output)


class MonotonicFusionEstimator:
    def __init__(self, checkpoint: dict[str, Any] | None = None) -> None:
        self.checkpoint = checkpoint or {"config": MonotonicFusionConfig().to_dict()}
        self.config = _config_from_checkpoint(self.checkpoint)

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "MonotonicFusionEstimator":
        checkpoint = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        return cls(checkpoint)

    def predict(self, *args: Any, **kwargs: Any) -> FusionResult:
        return monotonic_fuse(*args, config=self.config, **kwargs)


def _config_from_checkpoint(checkpoint: dict[str, Any]) -> MonotonicFusionConfig:
    payload = checkpoint.get("config", {})
    return MonotonicFusionConfig(**{key: payload.get(key, value) for key, value in MonotonicFusionConfig().__dict__.items()})
