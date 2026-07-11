from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .artifact_utils import read_json, relative_or_absolute, safe_float, write_csv, write_html_table, write_json


def build_measurement_artifacts(
    reports_path: str | Path,
    out_dir: str | Path,
    manifest_path: str | Path | None = None,
    frame_root: str | Path | None = None,
    measurement_dir: str | Path | None = None,
) -> dict[str, str]:
    reports = _load_reports(reports_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = read_json(manifest_path) if manifest_path else None
    root = Path(frame_root) if frame_root is not None else (Path(manifest_path).parent if manifest_path else Path("."))
    measurement_root = Path(measurement_dir) if measurement_dir else Path(reports_path).parent

    tables = {
        "measurement_summary_table": _summary_rows(reports),
        "lamp_status_table": _status_rows(reports),
        "affected_region_table": _affected_rows(reports),
        "illumination_feature_table": _feature_rows(reports),
        "attribution_table": _attribution_rows(reports),
        "calibration_abstention_table": _calibration_rows(reports),
        "physical_estimate_table": _physical_rows(reports),
        "measurement_flags_table": _flag_rows(reports),
    }
    artifacts: dict[str, str] = {}
    for name, rows in tables.items():
        artifacts[f"{name}_csv"] = str(write_csv(out / f"{name}.csv", rows))
    artifacts["measurement_summary_table_html"] = str(
        write_html_table(out / "measurement_summary_table.html", "Measurement Summary", tables["measurement_summary_table"])
    )
    route_rows = _route_rows(measurement_root)
    artifacts["route_aggregation_table_csv"] = str(write_csv(out / "route_aggregation_table.csv", route_rows))

    plot_dir = out / "measurement_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    artifacts.update(_write_measurement_plots(plot_dir, reports, route_rows))

    card_dir = out / "per_lamp_cards"
    artifacts["per_lamp_cards"] = str(_write_per_lamp_cards(card_dir, reports, manifest, root))
    contact = _measurement_contact_sheet(card_dir, out / "measurement_contact_sheet.jpg")
    if contact:
        artifacts["measurement_contact_sheet"] = str(contact)

    write_json(out / "measurement_artifacts_manifest.json", artifacts)
    return artifacts


def _load_reports(path: str | Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    if isinstance(payload, dict):
        payload = payload.get("reports", [])
    return [item for item in payload if isinstance(item, dict)]


def _summary_rows(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for report in reports:
        metrics = report.get("metrics", {}) or {}
        confidence = report.get("confidence", {}) or {}
        rows.append({
            "lamp_observation_id": report.get("lamp_observation_id"),
            "lamp_track_id": report.get("lamp_track_id"),
            "clip_id": report.get("clip_id"),
            "status": (report.get("status", {}) or {}).get("label"),
            "overall_category": metrics.get("overall_category"),
            "overall_score": metrics.get("overall_useful_illumination_score"),
            "confidence": confidence.get("overall"),
            "action": confidence.get("action"),
            "prediction_set": confidence.get("prediction_set"),
            "flags": ";".join(report.get("uncertainty_flags", []) or []),
            "physical_valid": (report.get("optional_physical_estimates", {}) or {}).get("valid"),
        })
    return rows


def _status_rows(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for report in reports:
        status = report.get("status", {}) or {}
        rows.append({
            "lamp_track_id": report.get("lamp_track_id"),
            "status_label": status.get("label"),
            "status_confidence": status.get("confidence"),
            "saturated_flag": status.get("saturated_flag"),
            "dim_probability": status.get("dim_probability"),
            "occluded_probability": status.get("occluded_probability"),
            "flicker_index": status.get("flicker_index"),
            "quality_flags": ";".join(status.get("quality_flags", []) or []),
        })
    return rows


def _affected_rows(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for report in reports:
        affected = report.get("affected_region", {}) or {}
        rows.append({
            "lamp_track_id": report.get("lamp_track_id"),
            "quality": affected.get("quality"),
            "geometry_quality": affected.get("geometry_quality"),
            "image_mask_uri": affected.get("image_mask_uri"),
            "region_mix": json.dumps(affected.get("region_mix", {}), ensure_ascii=True),
            "area_fraction": affected.get("area_fraction") or affected.get("support_fraction"),
        })
    return rows


def _feature_rows(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    names = [
        "coverage_proxy",
        "adequacy_proxy",
        "uniformity_proxy",
        "dark_hole_fraction",
        "glare_penalty",
        "occlusion_penalty",
        "temporal_stability",
        "confounder_penalty",
        "illumination_q10",
        "illumination_q50",
        "illumination_q90",
    ]
    rows = []
    for report in reports:
        metrics = report.get("metrics", {}) or {}
        row = {"lamp_track_id": report.get("lamp_track_id")}
        row.update({name: metrics.get(name) for name in names})
        rows.append(row)
    return rows


def _attribution_rows(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for report in reports:
        metrics = report.get("metrics", {}) or {}
        confidence = report.get("confidence", {}) or {}
        rows.append({
            "lamp_track_id": report.get("lamp_track_id"),
            "attribution_confidence": metrics.get("attribution_confidence") or metrics.get("attribution_score"),
            "attribution_class": confidence.get("attribution_class") or metrics.get("attribution_class"),
            "confounder_penalty": metrics.get("confounder_penalty"),
            "source_confusion_score": metrics.get("source_confusion_score"),
            "target_lamp_source": metrics.get("target_lamp_source"),
            "other_source": metrics.get("other_source"),
        })
    return rows


def _calibration_rows(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for report in reports:
        confidence = report.get("confidence", {}) or {}
        rows.append({
            "lamp_track_id": report.get("lamp_track_id"),
            "overall_confidence": confidence.get("overall"),
            "calibration_level": confidence.get("calibration_level"),
            "claim_tier": confidence.get("claim_tier"),
            "action": confidence.get("action"),
            "prediction_set": confidence.get("prediction_set"),
            "observation_completeness": confidence.get("observation_completeness"),
        })
    return rows


def _physical_rows(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for report in reports:
        physical = report.get("optional_physical_estimates", {}) or {}
        rows.append({
            "lamp_track_id": report.get("lamp_track_id"),
            "valid": physical.get("valid"),
            "reason": physical.get("reason"),
            "horizontal_illuminance_lux_mean": physical.get("horizontal_illuminance_lux_mean"),
            "horizontal_illuminance_lux_interval": physical.get("horizontal_illuminance_lux_interval"),
            "vertical_illuminance_lux_mean": physical.get("vertical_illuminance_lux_mean"),
            "served_area_m2_est": physical.get("served_area_m2_est"),
            "q_signal": physical.get("q_signal"),
            "q_calibration": physical.get("q_calibration"),
            "q_calib": physical.get("q_calib"),
        })
    return rows


def _flag_rows(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for report in reports:
        for flag in report.get("uncertainty_flags", []) or []:
            rows.append({"lamp_track_id": report.get("lamp_track_id"), "flag_type": "uncertainty", "flag": flag})
        physical = report.get("optional_physical_estimates", {}) or {}
        for flag in physical.get("quality_flags", []) or []:
            rows.append({"lamp_track_id": report.get("lamp_track_id"), "flag_type": "physical_quality", "flag": flag})
    return rows


def _route_rows(measurement_root: Path) -> list[dict[str, Any]]:
    route_path = measurement_root / "route_aggregation.json"
    if not route_path.exists():
        return []
    payload = read_json(route_path)
    rows = []
    for lamp in payload.get("lamps", []) or []:
        metrics = lamp.get("consensus_metrics", {}) or {}
        rows.append({
            "candidate_lamp_id": lamp.get("candidate_lamp_id"),
            "contributing_observations": ";".join(lamp.get("contributing_observations", []) or []),
            "overall_score": metrics.get("overall_useful_illumination_score"),
            "overall_category": metrics.get("overall_category"),
            "disagreement_score": lamp.get("disagreement_score"),
            "manual_review_priority": lamp.get("manual_review_priority"),
        })
    return rows


def _write_measurement_plots(out: Path, reports: list[dict[str, Any]], route_rows: list[dict[str, Any]]) -> dict[str, str]:
    metrics = [report.get("metrics", {}) or {} for report in reports]
    confidences = [report.get("confidence", {}) or {} for report in reports]
    scores = [safe_float(metric.get("overall_useful_illumination_score")) for metric in metrics]
    categories = [str(metric.get("overall_category", "unknown")) for metric in metrics]
    statuses = [str((report.get("status", {}) or {}).get("label", "unknown")) for report in reports]
    artifacts = {
        "measurement_score_distribution": str(_hist(out / "measurement_score_distribution.png", scores, "Useful-Illumination Score Distribution", "Score")),
        "measurement_category_counts": str(_bar_counts(out / "measurement_category_counts.png", Counter(categories), "Measurement Category Counts")),
        "status_counts": str(_bar_counts(out / "status_counts.png", Counter(statuses), "Lamp Status Counts")),
        "confidence_vs_score": str(_scatter(out / "confidence_vs_score.png", [safe_float(c.get("overall")) for c in confidences], scores, "Confidence vs Score", "Confidence", "Score")),
        "confounder_vs_score": str(_scatter(out / "confounder_vs_score.png", [safe_float(m.get("confounder_penalty")) for m in metrics], scores, "Confounder Penalty vs Score", "Confounder penalty", "Score")),
        "glare_occlusion_penalty": str(_scatter(out / "glare_occlusion_penalty.png", [safe_float(m.get("glare_penalty")) for m in metrics], [safe_float(m.get("occlusion_penalty")) for m in metrics], "Glare vs Occlusion Penalty", "Glare", "Occlusion")),
        "physical_validity_counts": str(_bar_counts(out / "physical_validity_counts.png", Counter(str((r.get("optional_physical_estimates", {}) or {}).get("valid")) for r in reports), "Physical Validity Counts")),
        "abstention_action_counts": str(_bar_counts(out / "abstention_action_counts.png", Counter(str(c.get("action", "report")) for c in confidences), "Abstention Action Counts")),
    }
    if route_rows:
        artifacts["route_candidate_lamp_scores"] = str(_bar_values(out / "route_candidate_lamp_scores.png", [str(r["candidate_lamp_id"]) for r in route_rows], [safe_float(r.get("overall_score")) for r in route_rows], "Route Candidate Lamp Scores"))
    return artifacts


def _hist(path: Path, values: list[float], title: str, xlabel: str) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(values or [0.0], bins=12, color="#74a9cf", edgecolor="#243b53")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _bar_counts(path: Path, counts: Counter[str], title: str) -> Path:
    return _bar_values(path, list(counts.keys()) or ["none"], list(counts.values()) or [0], title)


def _bar_values(path: Path, labels: list[str], values: list[float], title: str) -> Path:
    fig, ax = plt.subplots(figsize=(max(7, 0.35 * len(labels)), 4))
    ax.bar(range(len(labels)), values, color="#2ca25f")
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_title(title)
    ax.set_ylabel("Value")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _scatter(path: Path, x: list[float], y: list[float], title: str, xlabel: str, ylabel: str) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(x, y, color="#3182bd", alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _write_per_lamp_cards(out: Path, reports: list[dict[str, Any]], manifest: dict[str, Any] | None, frame_root: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    tracks_by_id: dict[str, list[dict[str, Any]]] = {}
    frames_by_id: dict[int, dict[str, Any]] = {}
    if manifest:
        for row in manifest.get("tracks", []) or []:
            tracks_by_id.setdefault(str(row.get("track_id")), []).append(row)
        frames_by_id = {int(frame.get("frame_id", 0)): frame for frame in manifest.get("frames", []) or []}
    for report in reports:
        track_id = str(report.get("lamp_track_id", "unknown"))
        png = out / f"{track_id}.png"
        _write_lamp_card_png(png, report, tracks_by_id.get(track_id, []), frames_by_id, frame_root)
        html = out / f"{track_id}.html"
        html.write_text(_lamp_card_html(track_id, report, png.name), encoding="utf-8")
    return out


def _write_lamp_card_png(path: Path, report: dict[str, Any], track_rows: list[dict[str, Any]], frames_by_id: dict[int, dict[str, Any]], frame_root: Path) -> None:
    card = Image.new("RGB", (760, 460), (245, 247, 250))
    draw = ImageDraw.Draw(card)
    font = _font(18)
    small = _font(13)
    track_id = str(report.get("lamp_track_id", "unknown"))
    if track_rows:
        row = sorted(track_rows, key=lambda item: int(item.get("frame_id", 0)))[0]
        frame = frames_by_id.get(int(row.get("frame_id", 0)))
        if frame:
            image_path = relative_or_absolute(Path(str(frame.get("image_uri"))), frame_root)
            if image_path.exists():
                image = Image.open(image_path).convert("RGB")
                bbox = [int(float(v)) for v in row.get("bbox_xyxy", [0, 0, 0, 0])]
                crop = image.crop((max(0, bbox[0] - 30), max(0, bbox[1] - 30), min(image.width, bbox[2] + 30), min(image.height, bbox[3] + 30)))
                crop.thumbnail((250, 180), Image.Resampling.LANCZOS)
                card.paste(crop, (24, 56))
    metrics = report.get("metrics", {}) or {}
    status = report.get("status", {}) or {}
    confidence = report.get("confidence", {}) or {}
    physical = report.get("optional_physical_estimates", {}) or {}
    draw.text((24, 20), f"Lamp Track {track_id}", fill=(20, 33, 61), font=font)
    lines = [
        f"Status: {status.get('label')} ({status.get('confidence', '')})",
        f"Category: {metrics.get('overall_category')} | Score: {metrics.get('overall_useful_illumination_score')}",
        f"Confidence/action: {confidence.get('overall')} / {confidence.get('action')}",
        f"Attribution: {confidence.get('attribution_class')} | {metrics.get('attribution_confidence')}",
        f"Confounder penalty: {metrics.get('confounder_penalty')} | Glare: {metrics.get('glare_penalty')}",
        f"Physical valid: {physical.get('valid')} | Reason: {physical.get('reason')}",
        f"Flags: {', '.join(report.get('uncertainty_flags', []) or [])}",
    ]
    y = 64
    for line in lines:
        draw.text((310, y), line[:95], fill=(20, 33, 61), font=small)
        y += 32
    card.save(path)


def _lamp_card_html(track_id: str, report: dict[str, Any], image_name: str) -> str:
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{track_id}</title></head>
<body style="font-family:Arial,sans-serif;margin:24px">
<h1>{track_id}</h1>
<img src="{image_name}" style="max-width:760px;width:100%">
<h2>Raw Report</h2>
<pre>{json.dumps(report, indent=2)}</pre>
</body></html>"""


def _measurement_contact_sheet(card_dir: Path, path: Path) -> Path | None:
    pngs = sorted(card_dir.glob("*.png"))[:16]
    if not pngs:
        return None
    thumbs = []
    for png in pngs:
        image = Image.open(png).convert("RGB")
        image.thumbnail((360, 220), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (360, 220), (30, 30, 30))
        canvas.paste(image, ((360 - image.width) // 2, (220 - image.height) // 2))
        thumbs.append(canvas)
    cols = 4
    rows = int(np.ceil(len(thumbs) / cols))
    sheet = Image.new("RGB", (cols * 360, rows * 220), (35, 35, 35))
    for idx, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((idx % cols) * 360, (idx // cols) * 220))
    sheet.save(path)
    return path


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build RBCCPS measurement presentation artifacts.")
    parser.add_argument("--reports", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--frame-root")
    parser.add_argument("--measurement-dir")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = build_measurement_artifacts(
        reports_path=args.reports,
        out_dir=args.out,
        manifest_path=args.manifest,
        frame_root=args.frame_root,
        measurement_dir=args.measurement_dir,
    )
    print(write_json(Path(args.out) / "run_output.json", artifacts).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
