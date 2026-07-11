from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CalibrationDecision:
    calibration_level: int
    proxy_allowed: bool
    physical_allowed: bool
    physical_reason: str
    claim_tier: str


class CalibrationPolicy:
    """Claim-tier policy for the measurement block.

    The policy deliberately separates proxy useful-illumination reporting from
    calibrated physical estimates. Ordinary video can produce proxy reports, but
    lux-like estimates require both sufficient calibration level and field
    reference evidence.
    """

    MIN_PHYSICAL_LEVEL = 3

    @classmethod
    def decide(
        cls,
        calibration_level: int,
        has_field_lux_calibration: bool,
        auto_exposure_active: bool,
        metadata_quality: str,
    ) -> CalibrationDecision:
        level = max(0, int(calibration_level))
        if level <= 0:
            tier = "Tier 0"
        elif level == 1:
            tier = "Tier 1"
        elif level == 2:
            tier = "Tier 2"
        else:
            tier = "Tier 3+"

        physical_allowed = (
            level >= cls.MIN_PHYSICAL_LEVEL
            and has_field_lux_calibration
            and not auto_exposure_active
            and metadata_quality in {"good", "complete", "controlled"}
        )
        if physical_allowed:
            reason = "Physical estimates allowed by calibration policy."
        elif level < cls.MIN_PHYSICAL_LEVEL:
            reason = "Calibration level is below the physical-estimate threshold."
        elif not has_field_lux_calibration:
            reason = "No field lux calibration is linked to this capture."
        elif auto_exposure_active:
            reason = "Auto exposure is active."
        else:
            reason = f"Metadata quality is insufficient: {metadata_quality}."

        return CalibrationDecision(
            calibration_level=level,
            proxy_allowed=True,
            physical_allowed=physical_allowed,
            physical_reason=reason,
            claim_tier=tier,
        )
