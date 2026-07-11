from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rbccps_measurement.photometry.sparse_reference_field import PhotometricBridgeConfig, load_lux_references_csv
from rbccps_measurement.photometry.torch_model import save_initialized_photometry_model, torch_available


@dataclass(frozen=True)
class PhotometryTrainingResult:
    checkpoint_json: Path
    weights_path: Path | None
    lux_rows: int
    valid_lux_rows: int
    status: str
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_json": str(self.checkpoint_json),
            "weights_path": str(self.weights_path) if self.weights_path else None,
            "lux_rows": self.lux_rows,
            "valid_lux_rows": self.valid_lux_rows,
            "status": self.status,
            "summary": self.summary,
        }


def train_photometry_module(dataset_root: str | Path, out: str | Path, config_path: str | Path | None = None) -> PhotometryTrainingResult:
    root = Path(dataset_root)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    config = _load_config(config_path)
    bridge_config = PhotometricBridgeConfig(**{key: config.get(key, value) for key, value in PhotometricBridgeConfig().__dict__.items()})
    lux_path = root / "lux" / "lux_points.csv"
    rows = _read_rows(lux_path)
    references = load_lux_references_csv(lux_path)
    valid_lux_rows = sum(1 for ref in references if ref.lux_value is not None and ref.validation_status not in {"invalid", "warning"})
    status = "initialized_not_optimized" if valid_lux_rows == 0 else "initialized_with_sparse_lux_references"

    weights_path: Path | None = None
    torch_summary: dict[str, Any] = {"torch_available": False, "trained_steps": 0}
    if torch_available():
        weights_path = out / "photometry_checkpoint.pt"
        torch_summary = save_initialized_photometry_model(weights_path)

    checkpoint = {
        "checkpoint_type": "module11_sparse_reference_photometry_checkpoint",
        "module": "photometry",
        "implementation": "deterministic_sparse_reference_photometric_field_v1",
        "dataset": str(root),
        "config": bridge_config.to_dict(),
        "weights": weights_path.name if weights_path else None,
        "status": status,
        "lux_rows": len(rows),
        "valid_lux_rows": valid_lux_rows,
        "losses_planned": ["isotonic_monotone_loss", "calibration_fit_loss", "uncertainty_calibration_nll_or_crps", "temporal_stability_loss"],
        "constraints": ["nonnegative_lux", "monotone_in_proxy_score", "uncertainty_increases_away_from_reference_support"],
        "assumptions": {
            "sparse_reference_interpretation": "sparse_reference_not_dense_ground_truth",
            "physical_claim": "approximate_lux_like_screening_not_certified_photometry",
            "vertical_estimates_require_vertical_reference_rows": True,
        },
        "training_summary": {
            "lux_points_csv": str(lux_path),
            "point_types": _count_values(rows, "point_type"),
            "orientation_counts": _count_values(rows, "orientation"),
            "torch": torch_summary,
        },
    }
    checkpoint_json = out / "photometry_checkpoint.json"
    checkpoint_json.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")
    return PhotometryTrainingResult(checkpoint_json, weights_path, len(rows), valid_lux_rows, status, checkpoint["training_summary"])


def _load_config(config_path: str | Path | None) -> dict[str, Any]:
    if not config_path:
        return {}
    return json.loads(Path(config_path).read_text(encoding="utf-8-sig"))


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _count_values(rows: list[dict[str, str]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(field) or "unknown")
        counts[value] = counts.get(value, 0) + 1
    return counts
