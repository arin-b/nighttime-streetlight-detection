from __future__ import annotations

from rbccps_measurement.contracts.input_schema import CalibrationRecord


def has_ground_projection_calibration(calibration: CalibrationRecord) -> bool:
    mount = calibration.mount or {}
    return mount.get("camera_height_m") is not None and mount.get("pitch_deg") is not None
