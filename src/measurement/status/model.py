from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rbccps_measurement.contracts.module_io import LampCropSequence, LatentEmissionStateOutput


STATUS_CLASSES = ("on", "dim", "off", "flicker", "occluded", "saturated", "unknown")
EMISSION_CLASSES = ("off", "dim", "on")
OCCLUSION_CLASSES = ("clear", "partial", "occluded")
CAPTURE_CLASSES = ("clean", "saturated", "blur_or_unreliable", "unknown")
FLICKER_CLASSES = ("stable", "possible_flicker", "flicker")


@dataclass(frozen=True)
class StatusModelConfig:
    sequence_length: int = 16
    crop_size: int = 64
    token_dim: int = 9
    hidden_dim: int = 64

    def to_dict(self) -> dict[str, int]:
        return {
            "sequence_length": self.sequence_length,
            "crop_size": self.crop_size,
            "token_dim": self.token_dim,
            "hidden_dim": self.hidden_dim,
        }


def _posterior(classes: tuple[str, ...], scores: dict[str, float]) -> dict[str, float]:
    raw = np.asarray([max(0.0, float(scores.get(label, 0.0))) for label in classes], dtype=np.float64)
    if float(np.sum(raw)) <= 0:
        raw[:] = 1.0 / len(classes)
    else:
        raw /= float(np.sum(raw))
    return {label: round(float(value), 6) for label, value in zip(classes, raw)}


def deterministic_latent_status(sequence: LampCropSequence) -> LatentEmissionStateOutput:
    valid = sequence.valid_mask.astype(bool)
    flags = list(sequence.quality_flags)
    if not np.any(valid):
        return LatentEmissionStateOutput(
            track_id=sequence.track_id,
            status_label="unknown",
            status_posterior=_posterior(STATUS_CLASSES, {"unknown": 1.0}),
            emission_posterior=_posterior(EMISSION_CLASSES, {"off": 0.2, "dim": 0.2, "on": 0.2}),
            occlusion_posterior=_posterior(OCCLUSION_CLASSES, {"occluded": 1.0}),
            capture_posterior=_posterior(CAPTURE_CLASSES, {"unknown": 1.0}),
            flicker_posterior=_posterior(FLICKER_CLASSES, {"stable": 1.0}),
            dim_probability=0.0,
            occluded_probability=1.0,
            flicker_index=0.0,
            saturated_flag=False,
            confidence=0.05,
            quality_flags=("empty_track_window",),
            metadata={"implementation": "deterministic_latent_status_v1"},
        )

    crops = sequence.crop_tensor[valid]
    tokens = sequence.metadata_tokens[valid]
    luma = 0.2126 * crops[:, :, :, 0] + 0.7152 * crops[:, :, :, 1] + 0.0722 * crops[:, :, :, 2]
    frame_luma = np.mean(luma, axis=(1, 2))
    peak_luma = np.quantile(luma, 0.98, axis=(1, 2))
    mean_luma = float(np.mean(frame_luma))
    peak = float(np.mean(peak_luma))
    temporal_var = float(np.std(frame_luma))

    saturation_fraction = float(np.mean(tokens[:, 1]))
    bloom_fraction = float(np.mean(tokens[:, 2]))
    glare_fraction = float(np.mean(tokens[:, 3]))
    reliability = float(np.mean(tokens[:, 4]))
    detector_score = float(np.mean(tokens[:, 5]))
    track_confidence = float(np.mean(tokens[:, 6]))
    lost_norm = float(np.mean(tokens[:, 7]))

    saturated_flag = saturation_fraction > 0.015 or glare_fraction > 0.08
    unsafe_saturation = saturation_fraction > 0.08 or (saturated_flag and reliability < 0.45)
    flicker_index = max(0.0, min(1.0, 5.0 * temporal_var + 0.65 * lost_norm))
    occluded_probability = max(0.0, min(1.0, 1.0 - track_confidence + 0.25 * (1.0 - reliability)))

    on_score = max(0.0, min(1.0, 0.45 * peak + 0.35 * detector_score + 0.20 * reliability))
    dim_score = max(0.0, min(1.0, 0.65 * (1.0 - peak) + 0.35 * detector_score)) if peak >= 0.12 else 0.25
    off_score = max(0.0, min(1.0, 1.0 - peak - 0.35 * detector_score)) if detector_score >= 0.25 else 0.05
    dim_probability = max(0.0, min(1.0, dim_score * (1.0 - saturation_fraction)))

    emission = _posterior(EMISSION_CLASSES, {"off": off_score, "dim": dim_probability, "on": on_score})
    occlusion = _posterior(
        OCCLUSION_CLASSES,
        {
            "clear": 1.0 - occluded_probability,
            "partial": max(0.0, 0.6 - abs(occluded_probability - 0.45)),
            "occluded": occluded_probability,
        },
    )
    capture = _posterior(
        CAPTURE_CLASSES,
        {
            "clean": reliability,
            "saturated": saturation_fraction + glare_fraction,
            "blur_or_unreliable": 1.0 - reliability,
            "unknown": 0.2 if "image_missing" in flags else 0.02,
        },
    )
    flicker = _posterior(
        FLICKER_CLASSES,
        {
            "stable": 1.0 - flicker_index,
            "possible_flicker": max(0.0, 0.7 - abs(flicker_index - 0.45)),
            "flicker": flicker_index,
        },
    )

    status_scores = {
        "on": on_score,
        "dim": dim_probability,
        "off": off_score,
        "flicker": flicker_index,
        "occluded": occluded_probability,
        "saturated": 1.0 if unsafe_saturation else 0.15 * float(saturated_flag),
        "unknown": max(0.05, 1.0 - reliability),
    }
    if unsafe_saturation:
        label = "saturated"
        flags.append("status_saturation_unsafe")
    elif flicker_index > 0.55:
        label = "flicker"
    elif occluded_probability > 0.68:
        label = "occluded"
    elif peak < 0.10 and detector_score >= 0.35 and reliability >= 0.45:
        label = "off"
    elif dim_probability > on_score and peak < 0.45:
        label = "dim"
    elif on_score >= 0.35:
        label = "on"
    else:
        label = "unknown"

    if saturated_flag:
        flags.append("saturated_flag")
    if bloom_fraction > 0.01:
        flags.append("bloom_evidence")
    if reliability < 0.45:
        flags.append("low_status_reliability")
    if "image_missing" in flags:
        label = "unknown"

    status = _posterior(STATUS_CLASSES, status_scores)
    confidence = max(0.05, min(1.0, 0.45 * status[label] + 0.35 * reliability + 0.20 * track_confidence))
    return LatentEmissionStateOutput(
        track_id=sequence.track_id,
        status_label=label,
        status_posterior=status,
        emission_posterior=emission,
        occlusion_posterior=occlusion,
        capture_posterior=capture,
        flicker_posterior=flicker,
        dim_probability=dim_probability,
        occluded_probability=occluded_probability,
        flicker_index=flicker_index,
        saturated_flag=saturated_flag,
        confidence=confidence,
        quality_flags=tuple(sorted(set(flags))),
        metadata={
            "implementation": "deterministic_latent_status_v1",
            "mean_luma": round(mean_luma, 6),
            "peak_luma": round(peak, 6),
            "temporal_luma_std": round(temporal_var, 6),
        },
    )


class StatusEstimator:
    def __init__(self, checkpoint: dict[str, Any] | None = None, torch_model: Any | None = None) -> None:
        self.checkpoint = checkpoint or {"implementation": "deterministic_latent_status_v1"}
        self.torch_model = torch_model

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "StatusEstimator":
        import json

        checkpoint_path = Path(path)
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8-sig"))
        weights_ref = checkpoint.get("weights")
        if weights_ref:
            weights_path = Path(weights_ref)
            if not weights_path.is_absolute():
                weights_path = checkpoint_path.parent / weights_path
            if weights_path.exists():
                try:
                    from rbccps_measurement.status.torch_model import load_torch_status_model

                    return cls(checkpoint=checkpoint, torch_model=load_torch_status_model(weights_path, checkpoint))
                except Exception as exc:
                    checkpoint["torch_load_error"] = str(exc)
        return cls(checkpoint=checkpoint)

    def predict(self, sequence: LampCropSequence) -> LatentEmissionStateOutput:
        if self.torch_model is not None:
            try:
                from rbccps_measurement.status.torch_model import predict_torch_status

                return predict_torch_status(self.torch_model, sequence)
            except Exception:
                pass
        return deterministic_latent_status(sequence)
