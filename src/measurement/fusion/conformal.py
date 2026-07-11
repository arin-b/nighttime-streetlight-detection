from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rbccps_measurement.contracts.module_io import CalibrationAbstentionOutput, SceneGraphFusionOutput


@dataclass(frozen=True)
class AbstentionDecision:
    action: str
    uncertainty_flags: list[str]
    prediction_set: list[str]
    calibration_output: CalibrationAbstentionOutput | None = None


@dataclass(frozen=True)
class CalibrationGroupKey:
    device_id: str
    route_group: str
    capture_mode: str
    confounder_density_bin: str
    gps_quality: str
    hdr_night_state: str

    def to_key(self) -> str:
        return "|".join(
            [
                f"device={self.device_id}",
                f"route={self.route_group}",
                f"mode={self.capture_mode}",
                f"conf={self.confounder_density_bin}",
                f"gps={self.gps_quality}",
                f"hdrnight={self.hdr_night_state}",
            ]
        )


@dataclass(frozen=True)
class ConformalCalibrationConfig:
    implementation: str = "deterministic_group_conformal_abstention_v1"
    confidence_floor: float = 0.35
    low_confidence_widen_threshold: float = 0.55
    abstain_risk_threshold: float = 0.68
    group_risk_weight: float = 0.22
    severe_flag_risk: float = 0.18

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def build_calibration_group_key(context: dict[str, Any] | None, confounder_density: float = 0.0) -> CalibrationGroupKey:
    context = context or {}
    if confounder_density >= 0.55:
        conf_bin = "high"
    elif confounder_density >= 0.25:
        conf_bin = "medium"
    else:
        conf_bin = "low"
    gps_quality = str(context.get("gps_quality") or "missing").lower()
    if gps_quality not in {"good", "partial", "missing", "poor"}:
        gps_quality = "partial"
    hdr = str(context.get("hdr_mode") or "unknown").lower()
    night = "night" if bool(context.get("night_mode")) else "notnight"
    return CalibrationGroupKey(
        device_id=str(context.get("device_id") or "unknown_device"),
        route_group=str(context.get("route_group") or "unknown_route"),
        capture_mode=str(context.get("capture_mode") or "night_video"),
        confounder_density_bin=conf_bin,
        gps_quality=gps_quality,
        hdr_night_state=f"{hdr}_{night}",
    )


def decide_abstention(
    overall_category: str,
    confidence: float,
    flags: list[str],
    fusion_output: SceneGraphFusionOutput | None = None,
    context: dict[str, Any] | None = None,
    physical_validity_score: float = 0.0,
    config: ConformalCalibrationConfig | None = None,
) -> AbstentionDecision:
    config = config or ConformalCalibrationConfig()
    context = context or {}
    uncertainty = fusion_output.uncertainty_index if fusion_output is not None else max(0.0, 1.0 - confidence)
    confounder_density = 0.0
    if fusion_output is not None:
        confounder_density = float(fusion_output.component_scores.get("confounder", 0.0))
    group_key = build_calibration_group_key(context, confounder_density)
    group_risk = _group_risk(group_key)
    severe_flags = [flag for flag in flags if flag in {"image_missing", "empty_affected_field", "high_fusion_uncertainty", "low_ris_confidence"}]
    risk = max(0.0, min(1.0, 1.0 - confidence + 0.38 * uncertainty + config.group_risk_weight * group_risk + config.severe_flag_risk * len(severe_flags)))
    calibrated_confidence = max(0.0, min(1.0, confidence * (1.0 - 0.35 * group_risk) * (1.0 - 0.25 * uncertainty)))
    prediction_set = _prediction_set(overall_category, calibrated_confidence, risk, config)
    out_flags = list(flags)
    if calibrated_confidence < config.low_confidence_widen_threshold:
        out_flags.append("low_conformal_confidence")
    if risk > config.abstain_risk_threshold:
        out_flags.append("high_group_conditioned_risk")
    action = "abstain" if risk > config.abstain_risk_threshold or calibrated_confidence < config.confidence_floor else "report"
    output = CalibrationAbstentionOutput(
        calibration_group_key=group_key.to_key(),
        calibrated_confidence=calibrated_confidence,
        prediction_set=tuple(prediction_set),
        risk_estimate=risk,
        abstention_action=action,
        physical_validity_score=max(0.0, min(1.0, float(physical_validity_score))),
        quality_flags=tuple(sorted(set(out_flags) - set(flags))),
        metadata={
            "implementation": config.implementation,
            "proxy_reporting_gate": "group_conditioned_risk",
            "physical_validity_gate": "separate_calibration_policy",
        },
    )
    return AbstentionDecision(action, sorted(set(out_flags)), prediction_set, calibration_output=output)


def _group_risk(group_key: CalibrationGroupKey) -> float:
    risk = 0.0
    if group_key.device_id == "unknown_device":
        risk += 0.15
    if group_key.route_group == "unknown_route":
        risk += 0.08
    if group_key.confounder_density_bin == "high":
        risk += 0.24
    elif group_key.confounder_density_bin == "medium":
        risk += 0.12
    if group_key.gps_quality in {"missing", "poor"}:
        risk += 0.12
    if "auto" in group_key.hdr_night_state or "unknown" in group_key.hdr_night_state:
        risk += 0.08
    return max(0.0, min(1.0, risk))


def _prediction_set(category: str, calibrated_confidence: float, risk: float, config: ConformalCalibrationConfig) -> list[str]:
    ordered = ["unknown", "poor", "marginal", "adequate"]
    if category not in ordered:
        category = "unknown"
    idx = ordered.index(category)
    labels = {category}
    if calibrated_confidence < config.low_confidence_widen_threshold or risk > 0.45:
        if idx > 0:
            labels.add(ordered[idx - 1])
        if idx < len(ordered) - 1:
            labels.add(ordered[idx + 1])
    if calibrated_confidence < config.confidence_floor or risk > config.abstain_risk_threshold:
        labels.add("manual_review_recommended")
    return sorted(labels, key=lambda item: ordered.index(item) if item in ordered else len(ordered))


class ConformalCalibrator:
    def __init__(self, checkpoint: dict[str, Any] | None = None) -> None:
        self.checkpoint = checkpoint or {"config": ConformalCalibrationConfig().to_dict()}
        self.config = _config_from_checkpoint(self.checkpoint)

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "ConformalCalibrator":
        checkpoint = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        return cls(checkpoint)

    def predict(self, *args: Any, **kwargs: Any) -> AbstentionDecision:
        return decide_abstention(*args, config=self.config, **kwargs)


def _config_from_checkpoint(checkpoint: dict[str, Any]) -> ConformalCalibrationConfig:
    payload = checkpoint.get("config", {})
    return ConformalCalibrationConfig(**{key: payload.get(key, value) for key, value in ConformalCalibrationConfig().__dict__.items()})
