#!/usr/bin/env python3
"""Plan, record, and review a non-destructive 20-image illumination-edit pilot.

Image editing is intentionally performed by the Codex image editing tool.  This
script owns deterministic selection, durable manifests, and contact sheets.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

ROOT = Path("datasets/raw/mobile_night_videos")
DEFAULT_RUN = Path("datasets/derived/counterfactual_streetlights_pilot_v1")
SEED = 20260912
STATES = ["off"] * 4 + ["flickering"] * 4 + ["mild_dim"] * 5 + ["moderate_severe_dim"] * 5 + ["mixed"] * 2
# Manually screened from the all-original preview; exactly 8/20 are scenes with
# substantial unrelated bright sources (traffic, signs, storefronts, etc.).
CONFOUNDER_RECORD_IDS = {"pilot_cf_001", "pilot_cf_003", "pilot_cf_004", "pilot_cf_005", "pilot_cf_008", "pilot_cf_009", "pilot_cf_011", "pilot_cf_019"}


def source_files() -> list[Path]:
    return sorted(p for i in range(1, 6) for p in (ROOT / f"synth_images_{i}").glob("*.jpg"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_plan(run_dir: Path) -> list[dict]:
    manifest = run_dir / "manifest.jsonl"
    if manifest.exists():
        records = [json.loads(line) for line in manifest.read_text().splitlines() if line]
        changed = False
        for record in records:
            wanted = record["record_id"] in CONFOUNDER_RECORD_IDS
            if record.get("confounder_priority") != wanted:
                record["confounder_priority"] = wanted
                changed = True
        if changed:
            manifest.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records))
        return records
    candidates = source_files()
    if len(candidates) < 20:
        raise RuntimeError(f"Expected at least 20 source images; found {len(candidates)}")
    rng = random.Random(SEED)
    selected = rng.sample(candidates, 20)
    states = STATES[:]
    rng.shuffle(states)
    now = datetime.now(timezone.utc).isoformat()
    records = []
    for index, (source, state) in enumerate(zip(selected, states), start=1):
        relative = source.relative_to(ROOT)
        output_name = f"{relative.parent.name}__{source.stem.replace(' ', '_').replace('(', '').replace(')', '')}__cf_{index:03d}__{state}.png"
        records.append({
            "record_id": f"pilot_cf_{index:03d}",
            "source_relative_path": str(relative),
            "source_filename": source.name,
            "source_sha256": sha256(source),
            "generated_relative_path": f"images/{output_name}",
            "transformation_type": state,
            "transformation_severity": {"off": "failed", "flickering": "intermediate", "mild_dim": "mild", "moderate_severe_dim": "moderate_or_severe", "mixed": "per_lamp"}[state],
            "confounder_priority": f"pilot_cf_{index:03d}" in CONFOUNDER_RECORD_IDS,
            "selection_seed": SEED,
            "generation_seed": None,
            "planned_timestamp_utc": now,
            "generation_timestamp_utc": None,
            "status": "planned",
            "attempt_count": 0,
            "model_request_id": None,
            "error": None,
            "qa_status": "pending",
        })
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "images").mkdir(exist_ok=True)
    (run_dir / "failed").mkdir(exist_ok=True)
    (run_dir / "run_config.json").write_text(json.dumps({"pilot_size": 20, "seed": SEED, "state_counts": dict(Counter(STATES)), "confounder_target_count": 8, "source_root": str(ROOT)}, indent=2) + "\n")
    manifest.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records))
    return records


def write_csv(run_dir: Path, records: list[dict]) -> None:
    with (run_dir / "manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def record_result(run_dir: Path, record_id: str, generated_file: Path | None, error: str | None) -> None:
    records = make_plan(run_dir)
    record = next((row for row in records if row["record_id"] == record_id), None)
    if record is None:
        raise RuntimeError(f"Unknown record id: {record_id}")
    record["attempt_count"] += 1
    record["generation_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    if generated_file:
        destination = run_dir / record["generated_relative_path"]
        if destination.exists():
            raise RuntimeError(f"Refusing to overwrite generated output: {destination}")
        shutil.copy2(generated_file, destination)
        record["status"] = "completed"
        record["error"] = None
    else:
        record["status"] = "failed"
        record["error"] = error or "image generation did not return an output"
    (run_dir / "manifest.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records))
    write_csv(run_dir, records)


def contact_sheet(run_dir: Path, records: list[dict]) -> None:
    tile_w, tile_h, header, gap = 320, 180, 26, 12
    canvas = Image.new("RGB", (tile_w * 2 + gap * 3, (tile_h + header + gap) * len(records) + gap), "#202124")
    draw = ImageDraw.Draw(canvas)
    for row_index, record in enumerate(records):
        y = gap + row_index * (tile_h + header + gap)
        source = ROOT / record["source_relative_path"]
        generated = run_dir / record["generated_relative_path"]
        for col, (title, path) in enumerate((("ORIGINAL", source), ("SYNTHETIC", generated))):
            x = gap + col * (tile_w + gap)
            with Image.open(path if path.exists() else source) as image:
                thumb = ImageOps.contain(image.convert("RGB"), (tile_w, tile_h))
            tile = Image.new("RGB", (tile_w, tile_h), "black")
            tile.paste(thumb, ((tile_w - thumb.width) // 2, (tile_h - thumb.height) // 2))
            canvas.paste(tile, (x, y + header))
            draw.text((x, y + 4), f"{record['record_id']}  {title}  {record['transformation_type']}", fill="white")
    canvas.save(run_dir / "contact_sheet.jpg", quality=92)


def normalize_outputs(run_dir: Path, records: list[dict]) -> None:
    """Enforce the source pixel dimensions without touching the source files."""
    for record in records:
        if record["status"] != "completed":
            continue
        source = ROOT / record["source_relative_path"]
        generated = run_dir / record["generated_relative_path"]
        with Image.open(source) as original, Image.open(generated) as edited:
            if edited.size != original.size:
                edited.convert("RGB").resize(original.size, Image.Resampling.LANCZOS).save(generated)
                record["output_dimensions"] = list(original.size)
                record["output_normalized_to_source_dimensions"] = True
            else:
                record["output_dimensions"] = list(edited.size)
                record["output_normalized_to_source_dimensions"] = False
    (run_dir / "manifest.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records))
    write_csv(run_dir, records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--contact-sheet", action="store_true")
    parser.add_argument("--normalize-outputs", action="store_true")
    parser.add_argument("--record-id")
    parser.add_argument("--generated-file", type=Path)
    parser.add_argument("--error")
    args = parser.parse_args()
    records = make_plan(args.run_dir)
    if args.record_id:
        if bool(args.generated_file) == bool(args.error):
            raise RuntimeError("Provide exactly one of --generated-file or --error with --record-id")
        record_result(args.run_dir, args.record_id, args.generated_file, args.error)
        records = make_plan(args.run_dir)
    if args.normalize_outputs:
        normalize_outputs(args.run_dir, records)
        records = make_plan(args.run_dir)
    write_csv(args.run_dir, records)
    if args.contact_sheet:
        contact_sheet(args.run_dir, records)
    print(json.dumps({"run_dir": str(args.run_dir), "records": len(records), "status_counts": dict(Counter(x["status"] for x in records)), "state_counts": dict(Counter(x["transformation_type"] for x in records))}))


if __name__ == "__main__":
    main()
