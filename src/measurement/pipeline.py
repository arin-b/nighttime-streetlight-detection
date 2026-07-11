from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from rbccps_measurement.attribution.counterfactual import estimate_counterfactual_attribution
from rbccps_measurement.contracts.calibration_policy import CalibrationPolicy
from rbccps_measurement.contracts.input_schema import LAMP_HEAD_CLASS, ClipManifest, DetectorTrackRecord
from rbccps_measurement.contracts.output_schema import MeasurementReport
from rbccps_measurement.decomposition.source_slots import estimate_source_evidence
from rbccps_measurement.decomposition.task_supervised_decomposition import deterministic_ris_decomposition
from rbccps_measurement.features.distributional_coverage import estimate_useful_features
from rbccps_measurement.fusion.conformal import decide_abstention
from rbccps_measurement.fusion.model_slot_fusion import fuse_model_slots
from rbccps_measurement.fusion.monotonic_heads import monotonic_fuse
from rbccps_measurement.geometry.lamp_footprint_field import estimate_footprint
from rbccps_measurement.ingest.validation import validate_clip_manifest
from rbccps_measurement.normalization.luma import estimate_normalization_quality
from rbccps_measurement.photometry.sparse_reference_field import estimate_photometric_field, physical_estimates_to_report
from rbccps_measurement.reporting.overlays import write_overlay_manifest
from rbccps_measurement.reporting.writers import write_csv, write_geojson, write_json
from rbccps_measurement.segmentation.illumination_disentangled import deterministic_segment_frame
from rbccps_measurement.status.latent_emission_state import estimate_lamp_status


def _mean(values: list[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def _group_tracks(tracks: tuple[DetectorTrackRecord, ...]) -> dict[str, list[DetectorTrackRecord]]:
    grouped: dict[str, list[DetectorTrackRecord]] = defaultdict(list)
    for track in tracks:
        if track.class_name != LAMP_HEAD_CLASS:
            continue
        grouped[track.track_id].append(track)
    for records in grouped.values():
        records.sort(key=lambda item: (item.timestamp_ns, item.frame_id))
    return dict(grouped)


class MeasurementPipeline:
    def __init__(
        self,
        measurement_run_id: str = "measurement_run",
        slot_metrics: dict[str, Any] | None = None,
        frame_root: str | Path = ".",
    ) -> None:
        self.measurement_run_id = measurement_run_id
        self.slot_metrics = slot_metrics
        self.frame_root = Path(frame_root)

    def run(self, manifest: ClipManifest) -> list[MeasurementReport]:
        validate_clip_manifest(manifest)
        frames = manifest.frame_by_id()
        grouped = _group_tracks(manifest.tracks)
        if not grouped:
            raise ValueError("measurement requires at least one streetlight_lamp_head track; pole-only manifests are support context, not lamp measurements")
        reports: list[MeasurementReport] = []

        for track_id, track_records in sorted(grouped.items()):
            frame_records = [frames[track.frame_id] for track in track_records]
            qualities = [estimate_normalization_quality(frame.camera) for frame in frame_records]
            normalization_reliability = sum(q.reliability for q in qualities) / len(qualities)
            flags = sorted({flag for quality in qualities for flag in quality.flags})

            if any((track.track_confidence or track.detector_score) < 0.65 for track in track_records):
                flags.append("moderate_detector_confidence")

            status = estimate_lamp_status(track_records, normalization_reliability)
            segmentation_by_frame = {
                frame.frame_id: deterministic_segment_frame(frame, self.frame_root)
                for frame in frame_records
            }
            footprint = estimate_footprint(track_id, track_records, frames, segmentation_by_frame)
            source_segmentation = segmentation_by_frame.get(footprint.field.frame_id) if footprint.field is not None else None
            source = estimate_source_evidence(
                track_records,
                frames,
                frame_root=self.frame_root,
                segmentation=source_segmentation,
                affected_region=footprint.field,
                status_confidence=status.confidence,
            )
            source_output = source.field_output
            ris_output = deterministic_ris_decomposition(
                frames[source_output.frame_id] if source_output is not None else frame_records[0],
                frame_root=self.frame_root,
                segmentation=source_segmentation,
                source_output=source_output,
            )
            if source_output is not None:
                flags.extend(source_output.quality_flags)
            flags.extend(ris_output.quality_flags)
            features = estimate_useful_features(
                status,
                footprint,
                source,
                normalization_reliability,
                segmentation=source_segmentation,
                ris_output=ris_output,
                source_output=source_output,
            )
            attribution = estimate_counterfactual_attribution(features, source, affected_region=footprint.field, source_output=source_output)

            observation_completeness = min(1.0, len(track_records) / max(1, max(track.track_age or len(track_records) for track in track_records)))
            camera_qualities = [frame.camera.metadata_quality for frame in frame_records]
            metadata_quality = "good" if all(q in {"good", "complete", "controlled"} for q in camera_qualities) else "partial"
            auto_exposure = any(frame.camera.auto_exposure_active for frame in frame_records)
            fusion_context = {
                "device_id": manifest.device_id,
                "route_group": manifest.optional_calibration.map_priors.get("route_group", "unknown_route"),
                "capture_mode": "night_video",
                "metadata_quality_score": 0.85 if metadata_quality == "good" else 0.55,
                "auto_exposure": 1.0 if auto_exposure else 0.0,
                "geometry_quality": footprint.geometry_quality,
                "gps_quality": "good" if all(frame.pose.gps_accuracy_m is not None and frame.pose.gps_accuracy_m <= 10 for frame in frame_records) else "missing",
                "hdr_mode": next((frame.camera.hdr_mode for frame in frame_records if frame.camera.hdr_mode), "unknown"),
                "night_mode": any(bool(frame.camera.night_mode) for frame in frame_records),
                "status_score": status.confidence,
            }
            fusion = monotonic_fuse(features, attribution, observation_completeness, track_id=track_id, region_mix=footprint.field.region_mix if footprint.field else {}, context=fusion_context)
            detector_score = max((track.track_confidence or track.detector_score) for track in track_records)
            slot_fusion = fuse_model_slots(fusion, self.slot_metrics, detector_score)
            fusion = slot_fusion.adjusted_result
            flags.extend(slot_fusion.flags)
            policy = CalibrationPolicy.decide(
                manifest.calibration_level,
                manifest.optional_calibration.has_field_lux_calibration,
                auto_exposure,
                metadata_quality,
            )
            decision = decide_abstention(
                fusion.overall_category,
                fusion.confidence,
                flags,
                fusion_output=fusion.fusion_output,
                context=fusion_context,
                physical_validity_score=1.0 if policy.physical_allowed else 0.0,
            )

            first_frame = frame_records[0]
            last_frame = frame_records[-1]
            region_mix = footprint.field.region_mix if footprint.field is not None else {}
            metrics = features.to_dict()
            metrics.update({
                "attribution_confidence": round(attribution.score, 4),
                **slot_fusion.metrics,
                "overall_useful_illumination_score": round(fusion.overall_score, 4),
                "overall_category": fusion.overall_category,
            })

            photometric_output = estimate_photometric_field(
                track_id=track_id,
                clip_id=manifest.clip_id,
                calibration=manifest.optional_calibration,
                policy=policy,
                useful_score=fusion.overall_score,
                fusion_confidence=fusion.confidence,
                glare_penalty=features.glare_penalty,
                dark_hole_fraction=features.dark_hole_fraction,
                confounder_penalty=features.confounder_penalty,
                geometry_quality=footprint.geometry_quality,
                metadata_quality=metadata_quality,
                auto_exposure_active=auto_exposure,
                conformal_risk=decision.calibration_output.risk_estimate if decision.calibration_output is not None else None,
                ris_confidence=ris_output.decomposition_confidence,
                source_confusion=source_output.source_confusion_score if source_output is not None else None,
                frame_records=frame_records,
            )
            physical = physical_estimates_to_report(photometric_output)

            report = MeasurementReport(
                measurement_run_id=self.measurement_run_id,
                lamp_observation_id=f"obs_{track_id}_{manifest.clip_id}",
                lamp_track_id=track_id,
                mapped_lamp_id=None,
                clip_id=manifest.clip_id,
                time_window={
                    "start_ns": first_frame.timestamp_ns,
                    "end_ns": last_frame.timestamp_ns,
                    "num_frames_used": len(track_records),
                    "evidence_frames": [track.frame_id for track in track_records[:4]],
                },
                geo_summary={
                    "lat": _mean([frame.pose.latitude for frame in frame_records]),
                    "lon": _mean([frame.pose.longitude for frame in frame_records]),
                    "gps_accuracy_m": _mean([frame.pose.gps_accuracy_m for frame in frame_records]),
                    "heading_deg": _mean([frame.pose.heading_deg for frame in frame_records]),
                },
                status=status.to_dict(),
                affected_region=footprint.to_dict(region_mix),
                metrics=metrics,
                confidence={
                    "overall": round(fusion.confidence, 4),
                    "calibration_level": policy.calibration_level,
                    "claim_tier": policy.claim_tier,
                    "observation_completeness": round(observation_completeness, 4),
                    "attribution_class": attribution.attribution_class,
                    "action": decision.action,
                    "prediction_set": decision.prediction_set,
                },
                uncertainty_flags=decision.uncertainty_flags,
                optional_physical_estimates=physical,
                traceability={
                    "model_versions": {
                        "pipeline": "deterministic_research_skeleton_v1",
                        "segmentation": "deterministic_illumination_disentangled_v1",
                        "footprint": "deterministic_lamp_conditioned_field_v1",
                        "source_decomposition": "deterministic_source_slots_v1",
                        "ris_decomposition": "deterministic_ris_decomposition_v1",
                        "features": "deterministic_distributional_features_v1",
                        "attribution": "deterministic_counterfactual_attribution_v1",
                        "trained_weights": self.slot_metrics.get("weights_used") if self.slot_metrics else None,
                        "fusion": "model_slot_fusion_v1" if self.slot_metrics else "deterministic_monotonic_scene_graph_fusion_v1",
                        "conformal": "deterministic_group_conformal_abstention_v1",
                        "photometry": "deterministic_sparse_reference_photometric_field_v1",
                    },
                    "feature_snapshot_ref": f"features/{track_id}_{manifest.clip_id}.json",
                    "policy_id": manifest.policy_id,
                },
            )
            reports.append(report)

        return reports


def _load_slot_metrics(manifest_path: Path, manifest: ClipManifest) -> dict[str, Any] | None:
    uri = manifest.optional_calibration.map_priors.get("learned_slot_metrics_uri")
    if not uri:
        return None
    path = Path(str(uri))
    if not path.is_absolute():
        path = manifest_path.parent / path
    return json.loads(path.read_text(encoding="utf-8-sig"))


def run_clip_to_directory(manifest_path: str | Path, output_dir: str | Path, measurement_run_id: str | None = None) -> list[MeasurementReport]:
    manifest_path = Path(manifest_path)
    manifest = ClipManifest.load(manifest_path)
    run_id = measurement_run_id or f"run_{manifest.clip_id}"
    reports = MeasurementPipeline(run_id, slot_metrics=_load_slot_metrics(manifest_path, manifest), frame_root=manifest_path.parent).run(manifest)
    out = Path(output_dir)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    (out / "features").mkdir(parents=True, exist_ok=True)
    write_json(out / "reports.json", reports)
    write_csv(out / "reports.csv", reports)
    write_geojson(out / "reports.geojson", reports)
    write_overlay_manifest(out / "overlays.json", reports)
    for report in reports:
        (out / report.affected_region["image_mask_uri"]).write_text(
            json.dumps({
                "lamp_track_id": report.lamp_track_id,
                "module": "lamp_conditioned_affected_region_field",
                "region_mix": report.affected_region["region_mix"],
                "geometry_quality": report.affected_region["geometry_quality"],
                "quality": report.affected_region["quality"],
                "note": "dense affected-field arrays are available inside Module 4 artifacts; JSON keeps report output lightweight",
            }, indent=2),
            encoding="utf-8",
        )
        feature_path = out / report.traceability["feature_snapshot_ref"]
        feature_path.write_text(json.dumps(report.metrics, indent=2), encoding="utf-8")
    return reports
