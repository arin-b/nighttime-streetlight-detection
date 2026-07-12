"""Multi-object tracker configuration writer (PDF §3).

Generates a YAML config file for BoT-SORT or ByteTrack that the
Ultralytics ``model.track()`` API consumes.
"""

from __future__ import annotations

from pathlib import Path

from evaluation.eval_pres.audit_config import TrackerConfig


def write_tracker_config(cfg: TrackerConfig, output_dir: Path) -> Path:
    """Write a tracker YAML to *output_dir* and return the path.

    Supports both ``botsort`` and ``ucmc`` tracker types.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if cfg.tracker_type == "ucmc":
        yaml_path = output_dir / "tracker_ucmc.yaml"
        # TODO: Implement full UCMC config generation once UCMC is integrated
        yaml_path.write_text("tracker_type: ucmc\n", encoding="utf-8")
    else:
        yaml_path = output_dir / "tracker_botsort.yaml"
        yaml_path.write_text(
            "\n".join([
                "tracker_type: botsort",
                f"track_high_thresh: {cfg.track_high_thresh}",
                f"track_low_thresh: {cfg.track_low_thresh}",
                f"new_track_thresh: {cfg.new_track_thresh}",
                f"track_buffer: {cfg.track_buffer}",
                f"match_thresh: {cfg.match_thresh}",
                "fuse_score: True",
                f"gmc_method: {cfg.gmc_method}",
                "proximity_thresh: 0.5",
                "appearance_thresh: 0.8",
                f"with_reid: {str(cfg.with_reid)}",
                "model: auto",
                "",
            ]),
            encoding="utf-8",
        )

    return yaml_path
