from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rbccps_measurement.contracts.input_schema import ClipManifest
from rbccps_measurement.decomposition.task_supervised_decomposition import RISDecompositionConfig, RISDecompositionEstimator, RIS_INTERPRETATION
from rbccps_measurement.decomposition.torch_models import save_initialized_ris_model, torch_available


@dataclass(frozen=True)
class RISDecompositionTrainingResult:
    checkpoint_json: Path
    checkpoint_weights: Path | None
    frames_seen: int
    status: str
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_json": str(self.checkpoint_json),
            "checkpoint_weights": str(self.checkpoint_weights) if self.checkpoint_weights else None,
            "frames_seen": self.frames_seen,
            "status": self.status,
            "summary": self.summary,
        }


def train_ris_decomposition_module(dataset_root: str | Path, out: str | Path, config_path: str | Path | None = None) -> RISDecompositionTrainingResult:
    root = Path(dataset_root)
    output = Path(out)
    output.mkdir(parents=True, exist_ok=True)
    config = _load_config(config_path)
    frames_seen = _count_frames(root)
    weights_path: Path | None = None
    summary: dict[str, Any] = {
        "module": "ris_decomposition",
        "architecture": "shared_encoder_reflectance_illumination_source_heads_v1",
        "frames_seen": frames_seen,
        "interpretation": RIS_INTERPRETATION,
        "losses_planned": [
            "reconstruction_loss",
            "decomposition_consistency_loss",
            "segmentation_aware_loss",
            "measurement_aware_loss",
            "source_separation_loss",
        ],
        "note": "Initialized representation only; this checkpoint must not be interpreted as calibrated physical Retinex.",
    }
    status = "initialized_no_torch"
    if torch_available():
        weights_path = output / "ris_decomposition_checkpoint.pt"
        summary.update(save_initialized_ris_model(weights_path))
        status = "initialized_no_dense_ris_labels"
    else:
        summary["torch_available"] = False

    checkpoint = {
        "checkpoint_type": "ris_decomposition_checkpoint",
        "module": "ris_decomposition",
        "status": status,
        "weights": weights_path.name if weights_path else None,
        "config": config.to_dict(),
        "label_maps": {
            "fields": ["reflectance_like", "illumination_like", "source_like", "confidence_map"],
        },
        "training_summary": summary,
        "fallback": "deterministic_ris_decomposition_v1",
    }
    checkpoint_json = output / "ris_decomposition_checkpoint.json"
    checkpoint_json.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")
    RISDecompositionEstimator.from_checkpoint(checkpoint_json)
    return RISDecompositionTrainingResult(checkpoint_json, weights_path, frames_seen, status, summary)


def _load_config(config_path: str | Path | None) -> RISDecompositionConfig:
    if not config_path:
        return RISDecompositionConfig()
    payload = json.loads(Path(config_path).read_text(encoding="utf-8-sig"))
    return RISDecompositionConfig(
        implementation=str(payload.get("implementation", "deterministic_ris_decomposition_v1")),
        blur_radius=int(payload.get("blur_radius", 5)),
        source_threshold=float(payload.get("source_threshold", 0.18)),
        min_confidence=float(payload.get("min_confidence", 0.05)),
        interpretation=str(payload.get("interpretation", RIS_INTERPRETATION)),
    )


def _count_frames(root: Path) -> int:
    manifest_path = root / "dataset_manifest.json"
    if not manifest_path.exists():
        return 0
    payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    total = 0
    for item in payload.get("clips", []):
        clip_ref = Path(item["manifest"])
        clip_path = clip_ref if clip_ref.is_absolute() else root / clip_ref
        if clip_path.exists():
            total += len(ClipManifest.load(clip_path).frames)
    return total
