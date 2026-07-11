from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from .artifact_utils import repo_root


DEFAULT_MODEL = Path("models/measurement/pretrained/streetlight_detector_v3/hpc_pull/best.pt")
DEFAULT_VIDEO = Path("eval_pres/sample_videos/busy_street_20260220_200501_629_1min.mp4")


def run_doctor(out: str | Path | None = None) -> dict[str, Any]:
    root = repo_root()
    checks = [
        _check_python(),
        _check_import("numpy", required=True),
        _check_import("matplotlib", required=True),
        _check_import("PIL", required=True, label="Pillow"),
        _check_import("sklearn", required=True, label="scikit-learn"),
        _check_import("cv2", required=True, label="opencv-python"),
        _check_import("ultralytics", required=True),
        _check_executable("ffmpeg", required=False),
        _check_path("default_detector", root / DEFAULT_MODEL, required=True),
        _check_path("bundled_sample_video", root / DEFAULT_VIDEO, required=True),
        _check_measurement_import(root),
        _check_write_access(root),
    ]
    status = "pass"
    if any(check["status"] == "fail" for check in checks):
        status = "fail"
    elif any(check["status"] == "warn" for check in checks):
        status = "warn"
    result = {
        "status": status,
        "python": sys.version,
        "platform": platform.platform(),
        "repo_root": str(root),
        "checks": checks,
    }
    if out is not None:
        out_path = Path(out)
        out_path.mkdir(parents=True, exist_ok=True)
        (out_path / "doctor_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _check_python() -> dict[str, Any]:
    ok = sys.version_info >= (3, 10)
    return {
        "name": "python_version",
        "status": "pass" if ok else "fail",
        "detail": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "required": ">=3.10",
    }


def _check_import(module: str, *, required: bool, label: str | None = None) -> dict[str, Any]:
    found = importlib.util.find_spec(module) is not None
    return {
        "name": label or module,
        "status": "pass" if found else ("fail" if required else "warn"),
        "detail": "importable" if found else "not importable",
        "required": required,
    }


def _check_executable(name: str, *, required: bool) -> dict[str, Any]:
    found = shutil.which(name)
    return {
        "name": name,
        "status": "pass" if found else ("fail" if required else "warn"),
        "detail": found or "not on PATH",
        "required": required,
    }


def _check_path(name: str, path: Path, *, required: bool) -> dict[str, Any]:
    exists = path.exists()
    return {
        "name": name,
        "status": "pass" if exists else ("fail" if required else "warn"),
        "detail": str(path),
        "required": required,
        "exists": exists,
        "size_mb": round(path.stat().st_size / (1024 * 1024), 3) if exists and path.is_file() else None,
    }


def _check_measurement_import(root: Path) -> dict[str, Any]:
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    try:
        import rbccps_measurement  # noqa: F401

        return {"name": "rbccps_measurement", "status": "pass", "detail": "importable from repo src", "required": True}
    except Exception as exc:
        return {"name": "rbccps_measurement", "status": "fail", "detail": str(exc), "required": True}


def _check_write_access(root: Path) -> dict[str, Any]:
    try:
        with tempfile.TemporaryDirectory(dir=root) as tmp:
            path = Path(tmp) / "write_check.txt"
            path.write_text("ok", encoding="utf-8")
        return {"name": "write_access", "status": "pass", "detail": str(root), "required": True}
    except Exception as exc:
        return {"name": "write_access", "status": "fail", "detail": str(exc), "required": True}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check eval_pres runtime dependencies and default assets.")
    parser.add_argument("--out", help="Optional directory for doctor_report.json.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_doctor(args.out)
    print(json.dumps(result, indent=2))
    if result["status"] == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
