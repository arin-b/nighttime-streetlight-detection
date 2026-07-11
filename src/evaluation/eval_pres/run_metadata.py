from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_run_sidecars(
    out: str | Path,
    *,
    run_type: str,
    parameters: dict[str, Any],
    summary: dict[str, Any] | None = None,
    argv: list[str] | None = None,
) -> dict[str, str]:
    out_path = Path(out)
    out_path.mkdir(parents=True, exist_ok=True)
    logs_dir = out_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    command = {
        "run_type": run_type,
        "argv": list(sys.argv if argv is None else argv),
        "parameters": _jsonable(parameters),
        "created_at_utc": utc_now_iso(),
    }
    environment = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
        "created_at_utc": utc_now_iso(),
    }
    run_summary = {
        "run_type": run_type,
        "status": "completed",
        "created_at_utc": utc_now_iso(),
        "summary": _jsonable(summary or {}),
    }

    paths = {
        "command": str(_write_json(out_path / "command.json", command)),
        "environment": str(_write_json(out_path / "environment.json", environment)),
        "run_summary": str(_write_json(out_path / "run_summary.json", run_summary)),
        "logs": str(logs_dir),
    }
    (logs_dir / "run.log").write_text(
        f"{utc_now_iso()} {run_type} completed; see command.json, environment.json, and run_summary.json\n",
        encoding="utf-8",
    )
    return paths


def _write_json(path: Path, payload: Any) -> Path:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
