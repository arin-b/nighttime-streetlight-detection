from pathlib import Path

from rbccps_measurement.models.downloader import cached_asset_path, ensure_required_assets
from rbccps_measurement.models.registry import get_registry


def test_required_model_assets_are_registered_for_dataflow():
    registry = get_registry()
    required = {name for name, spec in registry.items() if spec.required_for_training}
    assert {"streetlight_detector_v3", "segmentation_deeplabv3_mobilenet_v3", "feature_resnet18_imagenet"} <= required
    assert registry["segmentation_deeplabv3_mobilenet_v3"].module_stage.startswith("P_psi")
    assert registry["streetlight_detector_v3"].module_stage == "external_detector"


def test_required_model_assets_are_materialized():
    results = ensure_required_assets()
    for name, result in results.items():
        assert not result.startswith("unavailable:"), result
        assert Path(result).exists()
        assert cached_asset_path(name).exists()
