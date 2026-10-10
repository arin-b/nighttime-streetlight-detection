from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

from scripts import export_quantized_models as exporter


def test_export_report_contains_verified_absolute_model_path(tmp_path: Path, monkeypatch) -> None:
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"checkpoint")
    produced = tmp_path / "produced.onnx"

    class FakeYOLO:
        def __init__(self, path: str) -> None:
            assert path == str(weights)

        def export(self, **kwargs):
            assert kwargs["format"] == "onnx"
            produced.write_bytes(b"valid model contents")
            return produced

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYOLO))
    report = exporter.run_exports(weights, tmp_path / "exports", formats=["onnx_fp32"])

    assert report.status == "success"
    result = report.results[0]
    assert result.status == "success"
    assert Path(result.output_path).is_absolute()
    assert Path(result.output_path).read_bytes() == produced.read_bytes()
    saved = json.loads((tmp_path / "exports" / "export_report.json").read_text())
    assert saved["status"] == "success"
    assert saved["results"][0]["output_path"] == result.output_path


def test_missing_export_fails_cli_even_if_old_output_exists(tmp_path: Path, monkeypatch) -> None:
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"checkpoint")
    export_dir = tmp_path / "exports"
    stale = export_dir / "onnx_fp32" / "best.onnx"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"stale model")

    class FakeYOLO:
        def __init__(self, _path: str) -> None:
            pass

        def export(self, **_kwargs):
            return tmp_path / "nonexistent.onnx"

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYOLO))
    exit_code = exporter.main([
        "--weights", str(weights), "--out", str(export_dir), "--formats", "onnx_fp32",
    ])

    assert exit_code == 1
    saved = json.loads((export_dir / "export_report.json").read_text())
    assert saved["status"] == "error"
    assert saved["results"][0]["status"] == "error"
    assert saved["results"][0]["output_path"] == ""
    assert stale.read_bytes() == b"stale model"


def test_explicit_int8_without_calibration_is_not_success(tmp_path: Path, monkeypatch) -> None:
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"checkpoint")

    class FakeYOLO:
        def __init__(self, _path: str) -> None:
            pass

        def export(self, **_kwargs):
            raise AssertionError("INT8 export must be skipped before conversion")

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYOLO))
    exit_code = exporter.main([
        "--weights", str(weights), "--out", str(tmp_path / "exports"),
        "--formats", "tflite_int8",
    ])

    assert exit_code == 1
    saved = json.loads((tmp_path / "exports" / "export_report.json").read_text())
    assert saved["status"] == "error"
    assert saved["results"][0]["status"] == "skipped"
