from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any, Iterable


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path: str | Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    materialized = list(rows)
    if fieldnames is None:
        fieldnames = list(materialized[0].keys()) if materialized else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)
    return path


def write_html_table(path: str | Path, title: str, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    materialized = list(rows)
    if fieldnames is None:
        fieldnames = list(materialized[0].keys()) if materialized else []
    head = "".join(f"<th>{html.escape(str(name))}</th>" for name in fieldnames)
    body = []
    for row in materialized:
        cells = "".join(f"<td>{html.escape(_cell(row.get(name)))}</td>" for name in fieldnames)
        body.append(f"<tr>{cells}</tr>")
    doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #172033; }}
    h1 {{ font-size: 22px; margin-bottom: 12px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border: 1px solid #d7dde8; padding: 6px 8px; text-align: left; vertical-align: top; }}
    th {{ background: #eef2f7; position: sticky; top: 0; }}
    tr:nth-child(even) {{ background: #f9fbfd; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <table>
    <thead><tr>{head}</tr></thead>
    <tbody>{''.join(body)}</tbody>
  </table>
</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")
    return path


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=True)
    return str(value)


def flatten_dict(prefix: str, payload: dict[str, Any], out: dict[str, Any]) -> None:
    for key, value in payload.items():
        name = f"{prefix}_{key}" if prefix else str(key)
        if isinstance(value, dict):
            flatten_dict(name, value, out)
        else:
            out[name] = value


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def relative_or_absolute(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path
