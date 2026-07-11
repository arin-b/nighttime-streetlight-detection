from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from rbccps_measurement.contracts.calibration_policy import CalibrationDecision
from rbccps_measurement.contracts.input_schema import CalibrationRecord, FrameRecord
from rbccps_measurement.contracts.module_io import PhotometricFieldOutput, SparseLuxReference


IMPLEMENTATION = "deterministic_sparse_reference_photometric_field_v1"


@dataclass(frozen=True)
class PhotometricBridgeConfig:
    tau_required: float = 0.65
    min_lux: float = 0.01
    base_log_lux: float = math.log(0.6)
    score_log_lux_slope: float = 2.35
    calibration_log_lux_gain: float = 0.45
    low_reference_count: int = 3

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def physical_estimates_allowed(decision: CalibrationDecision) -> bool:
    return decision.physical_allowed


def parse_lux_references(rows: Iterable[dict[str, Any]], clip_id: str = "", track_id: str = "") -> tuple[SparseLuxReference, ...]:
    references: list[SparseLuxReference] = []
    for index, row in enumerate(rows):
        lux = _optional_float(row.get("lux_value", row.get("lux", row.get("value"))))
        references.append(
            SparseLuxReference(
                clip_id=str(row.get("clip_id") or clip_id),
                frame_id=str(row.get("frame_id") or ""),
                track_id=str(row.get("track_id") or track_id),
                point_type=str(row.get("point_type") or row.get("raw_label") or "unknown"),
                lux_value=lux,
                x=_optional_float(row.get("x")),
                y=_optional_float(row.get("y")),
                ground_x_m=_optional_float(row.get("ground_x_m", row.get("ground_x"))),
                ground_y_m=_optional_float(row.get("ground_y_m", row.get("ground_y"))),
                orientation=_normalize_orientation(row.get("orientation") or row.get("axis") or row.get("measurement_axis")),
                source_file=str(row.get("source_file") or ""),
                source_item=str(row.get("source_item") or f"lux_points[{index}]"),
                validation_status=str(row.get("validation_status") or "valid"),
                metadata={key: value for key, value in row.items() if key not in {"lux_value", "lux", "value"}},
            )
        )
    return tuple(references)


def load_lux_references_csv(path: str | Path, clip_id: str = "", track_id: str = "") -> tuple[SparseLuxReference, ...]:
    path = Path(path)
    if not path.exists():
        return ()
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return parse_lux_references(csv.DictReader(handle), clip_id=clip_id, track_id=track_id)


def references_from_calibration(calibration: CalibrationRecord, clip_id: str, track_id: str) -> tuple[SparseLuxReference, ...]:
    photometric = calibration.photometric or {}
    rows = photometric.get("lux_points") or photometric.get("sparse_lux_references") or []
    references = list(parse_lux_references(rows, clip_id=clip_id, track_id=track_id))
    calibration_id = photometric.get("field_lux_calibration_id")
    if calibration_id and not references:
        references.append(
            SparseLuxReference(
                clip_id=clip_id,
                frame_id="",
                track_id=track_id,
                point_type="linked_field_calibration",
                lux_value=None,
                orientation="horizontal",
                validation_status="linked_profile",
                metadata={"field_lux_calibration_id": calibration_id},
            )
        )
    return tuple(references)


def estimate_photometric_field(
    track_id: str,
    clip_id: str,
    calibration: CalibrationRecord,
    policy: CalibrationDecision,
    useful_score: float,
    fusion_confidence: float,
    glare_penalty: float,
    dark_hole_fraction: float,
    confounder_penalty: float,
    geometry_quality: float,
    metadata_quality: str,
    auto_exposure_active: bool,
    conformal_risk: float | None = None,
    ris_confidence: float | None = None,
    source_confusion: float | None = None,
    frame_records: Iterable[FrameRecord] = (),
    references: Iterable[SparseLuxReference] | None = None,
    config: PhotometricBridgeConfig | None = None,
) -> PhotometricFieldOutput:
    config = config or PhotometricBridgeConfig()
    refs = tuple(references) if references is not None else references_from_calibration(calibration, clip_id, track_id)
    explicit_refs = tuple(ref for ref in refs if ref.lux_value is not None and ref.validation_status not in {"invalid", "warning"})
    linked_reference = any(ref.validation_status == "linked_profile" for ref in refs)
    horizontal_refs = tuple(ref for ref in explicit_refs if ref.orientation == "horizontal")
    vertical_refs = tuple(ref for ref in explicit_refs if ref.orientation == "vertical")

    q_signal = _signal_quality(
        fusion_confidence=fusion_confidence,
        glare_penalty=glare_penalty,
        dark_hole_fraction=dark_hole_fraction,
        confounder_penalty=confounder_penalty,
        metadata_quality=metadata_quality,
        auto_exposure_active=auto_exposure_active,
        conformal_risk=conformal_risk,
        ris_confidence=ris_confidence,
        source_confusion=source_confusion,
    )
    q_calibration = _calibration_quality(
        calibration=calibration,
        policy=policy,
        geometry_quality=geometry_quality,
        frames=tuple(frame_records),
        explicit_reference_count=len(explicit_refs),
        linked_reference=linked_reference,
    )
    q_calib = min(q_signal, q_calibration)

    flags: list[str] = ["sparse_reference_not_dense_ground_truth"]
    if len(explicit_refs) < config.low_reference_count:
        flags.append("low_sparse_reference_count")
    if linked_reference and not explicit_refs:
        flags.append("linked_calibration_profile_without_inline_lux_rows")
    if vertical_refs:
        flags.append("vertical_reference_support_present")

    physical_valid = bool(policy.physical_allowed and q_calib >= config.tau_required)
    reason = _reason(policy, q_calib, config.tau_required, geometry_quality, refs)
    horizontal_mean: float | None = None
    horizontal_interval: tuple[float, float] | None = None
    served_area: float | None = None
    if physical_valid:
        horizontal_mean = _monotone_lux_estimate(useful_score, q_calib, horizontal_refs, config)
        half_width = _interval_half_width(
            lux=horizontal_mean,
            q_calib=q_calib,
            explicit_reference_count=len(explicit_refs),
            geometry_quality=geometry_quality,
            glare_penalty=glare_penalty,
            confounder_penalty=confounder_penalty,
            conformal_risk=conformal_risk,
            linked_reference=linked_reference,
        )
        horizontal_interval = (round(max(config.min_lux, horizontal_mean - half_width), 4), round(horizontal_mean + half_width, 4))
        horizontal_mean = round(horizontal_mean, 4)
        served_area = round(12.0 + 95.0 * _clip01(useful_score) * _clip01(geometry_quality), 4)

    vertical_mean = round(_mean_ref(vertical_refs), 4) if vertical_refs else None
    vertical_interval = _vertical_interval(vertical_refs, q_calib) if vertical_refs and physical_valid else None
    return PhotometricFieldOutput(
        track_id=track_id,
        clip_id=clip_id,
        q_signal=round(q_signal, 6),
        q_calibration=round(q_calibration, 6),
        q_calib=round(q_calib, 6),
        tau_required=config.tau_required,
        physical_valid=physical_valid,
        reason=reason if not physical_valid else "Physical estimates allowed by calibration and sparse-reference bridge.",
        horizontal_illuminance_lux_mean=horizontal_mean,
        horizontal_illuminance_lux_interval=horizontal_interval,
        vertical_illuminance_lux_mean=vertical_mean if vertical_refs else None,
        vertical_illuminance_lux_interval=vertical_interval,
        served_area_m2_est=served_area,
        reference_summary={
            "total_references": len(refs),
            "explicit_lux_references": len(explicit_refs),
            "horizontal_references": len(horizontal_refs),
            "vertical_references": len(vertical_refs),
            "linked_field_lux_calibration": linked_reference,
        },
        quality_flags=tuple(sorted(set(flags))),
        metadata={
            "implementation": IMPLEMENTATION,
            "claim": "approximate_lux_like_screening_not_certified_photometry",
            "mapping": "monotone_log_lux_proxy_mapping",
            "constraints": ["nonnegative_lux", "monotone_in_proxy_score", "uncertainty_increases_when_reference_support_weakens"],
            "sparse_reference_interpretation": "sparse_reference_not_dense_ground_truth",
        },
    )


def physical_estimates_to_report(output: PhotometricFieldOutput) -> dict[str, Any]:
    return {
        "valid": output.physical_valid,
        "reason": output.reason,
        "horizontal_illuminance_lux_mean": output.horizontal_illuminance_lux_mean,
        "horizontal_illuminance_lux_interval": list(output.horizontal_illuminance_lux_interval) if output.horizontal_illuminance_lux_interval else None,
        "vertical_illuminance_lux_mean": output.vertical_illuminance_lux_mean,
        "served_area_m2_est": output.served_area_m2_est,
        "q_signal": output.q_signal,
        "q_calibration": output.q_calibration,
        "q_calib": output.q_calib,
        "tau_required": output.tau_required,
        "reference_summary": output.reference_summary,
        "quality_flags": list(output.quality_flags),
    }


def _signal_quality(
    fusion_confidence: float,
    glare_penalty: float,
    dark_hole_fraction: float,
    confounder_penalty: float,
    metadata_quality: str,
    auto_exposure_active: bool,
    conformal_risk: float | None,
    ris_confidence: float | None,
    source_confusion: float | None,
) -> float:
    metadata_score = {"good": 0.92, "complete": 0.95, "controlled": 1.0, "partial": 0.62, "pseudo": 0.45, "missing": 0.25}.get(str(metadata_quality).lower(), 0.5)
    exposure_score = 0.35 if auto_exposure_active else 1.0
    confidence_floor = 0.65 if metadata_score >= 0.9 and not auto_exposure_active else 0.0
    confidence_support = max(_clip01(fusion_confidence), confidence_floor)
    ris_score = _clip01(0.7 if ris_confidence is None else ris_confidence)
    risk_score = 1.0 - _clip01(0.35 if conformal_risk is None else conformal_risk)
    source_score = 1.0 - _clip01(0.15 if source_confusion is None else source_confusion)
    return _clip01(
        0.24 * metadata_score
        + 0.25 * exposure_score
        + 0.10 * confidence_support
        + 0.16 * (1.0 - _clip01(glare_penalty))
        + 0.04 * (1.0 - _clip01(dark_hole_fraction))
        + 0.08 * (1.0 - _clip01(confounder_penalty))
        + 0.06 * ris_score
        + 0.04 * risk_score
        + 0.03 * source_score
    )


def _calibration_quality(
    calibration: CalibrationRecord,
    policy: CalibrationDecision,
    geometry_quality: float,
    frames: tuple[FrameRecord, ...],
    explicit_reference_count: int,
    linked_reference: bool,
) -> float:
    photometric = calibration.photometric or {}
    level_score = _clip01(policy.calibration_level / 3.0)
    field_link_score = 1.0 if calibration.has_field_lux_calibration else 0.0
    reference_score = min(1.0, explicit_reference_count / 5.0)
    if linked_reference and reference_score == 0.0:
        reference_score = 0.70
    response_score = _clip01(_optional_float(photometric.get("response_curve_quality"), 0.75 if calibration.has_field_lux_calibration else 0.25))
    vignetting_score = _clip01(_optional_float(photometric.get("vignetting_quality"), 0.70 if calibration.has_field_lux_calibration else 0.25))
    gps_score = _gps_quality(frames)
    return _clip01(
        0.25 * level_score
        + 0.18 * field_link_score
        + 0.18 * _clip01(geometry_quality)
        + 0.17 * reference_score
        + 0.10 * response_score
        + 0.07 * vignetting_score
        + 0.05 * gps_score
    )


def _reason(policy: CalibrationDecision, q_calib: float, tau_required: float, geometry_quality: float, refs: tuple[SparseLuxReference, ...]) -> str:
    if not policy.physical_allowed:
        return policy.physical_reason
    if not refs:
        return "No sparse lux reference or linked field calibration is available."
    if geometry_quality <= 0.0:
        return "Ground-plane or affected-region geometry is unavailable."
    if q_calib < tau_required:
        return f"Module 11 calibration quality {q_calib:.3f} is below required threshold {tau_required:.3f}."
    return "Physical estimates allowed by calibration and sparse-reference bridge."


def _monotone_lux_estimate(score: float, q_calib: float, horizontal_refs: tuple[SparseLuxReference, ...], config: PhotometricBridgeConfig) -> float:
    score = _clip01(score)
    ref_lux = _mean_ref(horizontal_refs)
    base_log_lux = math.log(max(config.min_lux, ref_lux)) if ref_lux is not None else config.base_log_lux
    centered_score = score - 0.5
    log_lux = base_log_lux + config.score_log_lux_slope * centered_score + config.calibration_log_lux_gain * (_clip01(q_calib) - 0.65)
    return max(config.min_lux, math.exp(log_lux))


def _interval_half_width(
    lux: float,
    q_calib: float,
    explicit_reference_count: int,
    geometry_quality: float,
    glare_penalty: float,
    confounder_penalty: float,
    conformal_risk: float | None,
    linked_reference: bool,
) -> float:
    sparsity = 0.35 if explicit_reference_count >= 5 else 0.75 if explicit_reference_count > 0 else 1.05
    if linked_reference and explicit_reference_count == 0:
        sparsity = 0.85
    risk = _clip01(0.35 if conformal_risk is None else conformal_risk)
    relative = 0.18 + 0.45 * (1.0 - _clip01(q_calib)) + 0.20 * (1.0 - _clip01(geometry_quality)) + 0.14 * _clip01(glare_penalty) + 0.12 * _clip01(confounder_penalty) + 0.12 * risk + 0.16 * sparsity
    return max(0.15, lux * relative)


def _vertical_interval(vertical_refs: tuple[SparseLuxReference, ...], q_calib: float) -> tuple[float, float] | None:
    mean = _mean_ref(vertical_refs)
    if mean is None:
        return None
    half = max(0.15, mean * (0.25 + 0.45 * (1.0 - _clip01(q_calib))))
    return (round(max(0.01, mean - half), 4), round(mean + half, 4))


def _gps_quality(frames: tuple[FrameRecord, ...]) -> float:
    if not frames:
        return 0.55
    scores = []
    for frame in frames:
        accuracy = frame.pose.gps_accuracy_m
        if accuracy is None:
            scores.append(0.35)
        elif accuracy <= 5:
            scores.append(1.0)
        elif accuracy <= 10:
            scores.append(0.78)
        elif accuracy <= 25:
            scores.append(0.45)
        else:
            scores.append(0.25)
    return sum(scores) / len(scores)


def _mean_ref(refs: tuple[SparseLuxReference, ...]) -> float | None:
    values = [float(ref.lux_value) for ref in refs if ref.lux_value is not None and ref.lux_value > 0]
    if not values:
        return None
    return sum(values) / len(values)


def _optional_float(value: object, default: float | None = None) -> float | None:
    if value in {None, ""}:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_orientation(value: object) -> str:
    raw = str(value or "horizontal").lower()
    if raw.startswith("vert") or raw in {"ev", "vertical_lux"}:
        return "vertical"
    if raw.startswith("horiz") or raw in {"eh", "lux", "horizontal_lux"}:
        return "horizontal"
    return "unknown"


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
