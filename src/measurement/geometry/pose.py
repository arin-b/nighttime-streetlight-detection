from __future__ import annotations

from rbccps_measurement.contracts.input_schema import PoseRecord


def gps_is_usable(pose: PoseRecord, max_accuracy_m: float = 10.0) -> bool:
    return pose.latitude is not None and pose.longitude is not None and (pose.gps_accuracy_m or 9999.0) <= max_accuracy_m
