from __future__ import annotations

from rbccps_measurement.contracts.input_schema import CameraMetadata


def exposure_iso_product(camera: CameraMetadata) -> float | None:
    if camera.exposure_time_s is None or camera.sensor_sensitivity_iso is None:
        return None
    return float(camera.exposure_time_s) * float(camera.sensor_sensitivity_iso)
