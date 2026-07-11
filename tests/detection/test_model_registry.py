from rbccps_od.models.registry import get_asset, get_registry


def test_registry_contains_required_assets():
    registry = get_registry()
    assert "yolov26_base" in registry
    assert "lowlight_enhancer" in registry


def test_advanced_assets_are_pinned_to_exact_upstream_sources():
    spec = get_asset("lowlight_enhancer")
    assert spec.url == "https://github.com/Li-Chongyi/Zero-DCE/archive/refs/heads/master.zip"
    assert spec.source_repo_url == "https://github.com/Li-Chongyi/Zero-DCE"
    assert spec.implementation == "source_archive_only"


def test_runtime_checkpoint_assets_are_pinned_to_archive_members():
    spec = get_asset("zero_dce_epoch99")
    assert spec.implementation == "archive_member"
    assert spec.metadata["archive_asset"] == "lowlight_enhancer"
    assert spec.metadata["archive_member"].endswith("Epoch99.pth")
