from pathlib import Path

from detection.config.hyperparams import load_detection_hyperparameters
from detection.pipeline.multicue_stage import MultiCueFilterStage


def test_load_detection_hyperparameters_merges_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "custom_detection.yaml"
    config_path.write_text(
        """
detector:
  conf_threshold: 0.42
  use_geometry_attention: true
  geometry_attention:
    kernel_size: 9
    neutral_bias: 3.5
  negative_attention:
    reduction: 2
    init_bias: -2.0
multicue:
  enabled: true
  aggregation_threshold: 0.66
  cue_weights:
    trajectory: 0.1
    size_progression: 0.2
    light_characteristics: 0.3
    position_prior: 0.4
  aspect_ratio_min: 0.2
  aspect_ratio_max: 2.0
  spatial_upper_fraction: 0.9
  min_brightness_for_detection: 15
  min_track_frames_for_confirmation: 2
  duplicate_center_distance_px: 10.0
""",
        encoding="utf-8",
    )

    payload = load_detection_hyperparameters(config_path)

    assert payload["detector"]["conf_threshold"] == 0.42
    assert payload["detector"]["geometry_attention"]["kernel_size"] == 9
    assert payload["detector"]["negative_attention"]["reduction"] == 2
    assert payload["multicue"]["aggregation_threshold"] == 0.66


def test_multicue_stage_uses_yaml_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "custom_detection.yaml"
    config_path.write_text(
        """
multicue:
  enabled: true
  aggregation_threshold: 0.6
  cue_weights:
    trajectory: 0.1
    size_progression: 0.2
    light_characteristics: 0.3
    position_prior: 0.4
""",
        encoding="utf-8",
    )

    stage = MultiCueFilterStage(config_path=config_path)

    assert stage.enabled is True
    assert stage.threshold == 0.6
    assert stage.weights.trajectory == 0.1
    assert stage.weights.position_prior == 0.4
