#!/usr/bin/env python3
"""
Export a fine-tuned YOLO model to multiple quantized formats for mobile deployment.

Post-training quantization only — no QAT required.

Usage
-----
    python scripts/export_quantized_models.py \
        --weights /path/to/best.pt \
        --out runs/quantized_exports \
        --imgsz 640 \
        --data /path/to/dataset.yaml          # optional, needed for INT8 calibration
        --formats onnx_fp32 onnx_fp16 tflite_fp32 tflite_fp16 tflite_int8 ncnn torchscript coreml

Supported export targets
------------------------
    onnx_fp32       ONNX (full precision)         — baseline, any platform
    onnx_fp16       ONNX (float16)                — smaller, GPU-capable mobile
    tflite_fp32     TFLite (full precision)        — Android / iOS baseline
    tflite_fp16     TFLite (float16)               — Android / iOS, ~2× smaller
    tflite_int8     TFLite (int8, needs --data)    — fastest on-device, smallest
    ncnn            NCNN (Tencent)                 — very fast Android/Linux edge
    torchscript     TorchScript                    — PyTorch Mobile
    coreml          CoreML (FP16 by default)       — iOS / macOS
    engine_fp16     TensorRT FP16                  — NVIDIA GPU (Jetson, desktop)
    engine_int8     TensorRT INT8 (needs --data)   — NVIDIA GPU, maximum speed

Notes
-----
- INT8 exports require a calibration dataset via --data (your dataset.yaml).
- TensorRT exports require an NVIDIA GPU with TensorRT installed.
- CoreML exports may require coremltools (`pip install coremltools`).
- NCNN exports require ncnn converter (`pip install ncnn`).
  If a dependency is missing the script will skip that format and continue.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# ── All supported export configurations ──────────────────────────────────────

ALL_FORMATS = [
    "onnx_fp32",
    "onnx_fp16",
    "tflite_fp32",
    "tflite_fp16",
    "tflite_int8",
    "ncnn",
    "torchscript",
    "coreml",
    "engine_fp16",
    "engine_int8",
]

# Mobile-first default subset (no TensorRT, no CoreML)
MOBILE_DEFAULTS = [
    "onnx_fp32",
    "onnx_fp16",
    "tflite_fp32",
    "tflite_fp16",
    "tflite_int8",
    "ncnn",
    "torchscript",
]


@dataclass
class ExportResult:
    format_name: str
    ultralytics_format: str
    precision: str
    status: str                     # "success" | "skipped" | "error"
    export_time_s: float = 0.0
    output_path: str = ""
    file_size_mb: float = 0.0
    error_message: str = ""


@dataclass
class ExportReport:
    source_weights: str
    image_size: int
    calibration_data: str | None
    timestamp: str = ""
    results: list[ExportResult] = field(default_factory=list)
    status: str = "planned"


# ── Export dispatch table ────────────────────────────────────────────────────

def _export_one(
    model,
    fmt: str,
    out_dir: Path,
    imgsz: int,
    data: str | None,
) -> ExportResult:
    """Run a single export and return the result record."""

    # Map our format names → ultralytics export() kwargs
    DISPATCH: dict[str, dict[str, Any]] = {
        "onnx_fp32": dict(format="onnx", half=False),
        "onnx_fp16": dict(format="onnx", half=True),
        "tflite_fp32": dict(format="tflite", half=False, int8=False),
        "tflite_fp16": dict(format="tflite", half=True, int8=False),
        "tflite_int8": dict(format="tflite", int8=True),
        "ncnn": dict(format="ncnn"),
        "torchscript": dict(format="torchscript"),
        "coreml": dict(format="coreml"),
        "engine_fp16": dict(format="engine", half=True),
        "engine_int8": dict(format="engine", int8=True),
    }

    precision_labels = {
        "onnx_fp32": "FP32",
        "onnx_fp16": "FP16",
        "tflite_fp32": "FP32",
        "tflite_fp16": "FP16",
        "tflite_int8": "INT8",
        "ncnn": "FP32",
        "torchscript": "FP32",
        "coreml": "FP16",
        "engine_fp16": "FP16",
        "engine_int8": "INT8",
    }

    if fmt not in DISPATCH:
        return ExportResult(
            format_name=fmt,
            ultralytics_format="?",
            precision="?",
            status="error",
            error_message=f"Unknown format {fmt!r}. Choose from: {', '.join(ALL_FORMATS)}",
        )

    kwargs = DISPATCH[fmt].copy()
    ul_format = kwargs["format"]
    precision = precision_labels[fmt]

    # INT8 formats require a calibration dataset
    if kwargs.get("int8") and not data:
        return ExportResult(
            format_name=fmt,
            ultralytics_format=ul_format,
            precision=precision,
            status="skipped",
            error_message="INT8 calibration requires --data <dataset.yaml>; skipping.",
        )
    if kwargs.get("int8") and data:
        kwargs["data"] = data

    t0 = time.perf_counter()
    try:
        exported_path = model.export(imgsz=imgsz, **kwargs)
    except Exception as exc:
        return ExportResult(
            format_name=fmt,
            ultralytics_format=ul_format,
            precision=precision,
            status="error",
            export_time_s=round(time.perf_counter() - t0, 2),
            error_message=str(exc),
        )
    elapsed = round(time.perf_counter() - t0, 2)

    # Verify the returned artifact before recording a successful export. A stale
    # file in the output directory must never stand in for a failed conversion.
    exported = Path(str(exported_path)).expanduser().resolve()
    expected_suffixes = {
        "onnx_fp32": ".onnx", "onnx_fp16": ".onnx",
        "tflite_fp32": ".tflite", "tflite_fp16": ".tflite", "tflite_int8": ".tflite",
        "torchscript": ".torchscript", "engine_fp16": ".engine", "engine_int8": ".engine",
    }
    if fmt == "ncnn":
        valid = exported.is_dir() and bool(list(exported.rglob("*.param"))) and bool(list(exported.rglob("*.bin")))
    elif fmt == "coreml":
        valid = (exported.is_dir() and exported.suffix == ".mlpackage") or (exported.is_file() and exported.suffix == ".mlmodel")
    else:
        valid = exported.is_file() and exported.suffix == expected_suffixes[fmt]
    if not valid:
        return ExportResult(
            format_name=fmt, ultralytics_format=ul_format, precision=precision,
            status="error", export_time_s=elapsed,
            error_message=f"Export did not create the expected {fmt} model at {exported}",
        )

    dest = out_dir / fmt
    if exported in dest.parents:
        return ExportResult(
            format_name=fmt, ultralytics_format=ul_format, precision=precision,
            status="error", export_time_s=elapsed,
            error_message=f"Export destination is nested inside source model: {dest}",
        )
    dest.mkdir(parents=True, exist_ok=True)
    if exported == dest or dest in exported.parents:
        target = exported
    else:
        target = dest / exported.name
        if target.exists() or target.is_symlink():
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            else:
                target.unlink()
        if exported.is_dir():
            shutil.copytree(exported, target)
        else:
            shutil.copy2(exported, target)
    size_bytes = target.stat().st_size if target.is_file() else sum(
        item.stat().st_size for item in target.rglob("*") if item.is_file()
    )
    if size_bytes <= 0:
        return ExportResult(
            format_name=fmt, ultralytics_format=ul_format, precision=precision,
            status="error", export_time_s=elapsed,
            error_message=f"Exported model is empty: {target}",
        )
    if fmt == "onnx_fp16":
        import onnx
        graph = onnx.load(str(target), load_external_data=False).graph
        if not any(item.data_type == onnx.TensorProto.FLOAT16 for item in graph.initializer):
            return ExportResult(
                format_name=fmt, ultralytics_format=ul_format, precision=precision,
                status="error", export_time_s=elapsed,
                error_message=f"ONNX FP16 conversion produced no FP16 weights: {target}",
            )

    return ExportResult(
        format_name=fmt,
        ultralytics_format=ul_format,
        precision=precision,
        status="success",
        export_time_s=elapsed,
        output_path=str(target.resolve()),
        file_size_mb=round(size_bytes / (1024 * 1024), 2),
    )


# ── Main ─────────────────────────────────────────────────────────────────────

def run_exports(
    weights: str | Path,
    out: str | Path,
    imgsz: int = 640,
    data: str | None = None,
    formats: list[str] | None = None,
) -> ExportReport:
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics is not installed. Run: pip install ultralytics", file=sys.stderr)
        sys.exit(1)

    weights_path = Path(weights).expanduser().resolve()
    if not weights_path.is_file():
        print(f"ERROR: weights file not found: {weights_path}", file=sys.stderr)
        sys.exit(1)
    if data:
        data_path = Path(data).expanduser().resolve()
        if not data_path.is_file():
            print(f"ERROR: calibration dataset YAML not found: {data_path}", file=sys.stderr)
            sys.exit(1)
        data = str(data_path)

    # Direct script execution from a source checkout must resolve the custom
    # module classes stored in attention-ablation PyTorch checkpoints.
    src_dir = str(Path(__file__).resolve().parent.parent / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    out_dir = Path(out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    formats_to_export = list(formats) if formats is not None else [
        fmt for fmt in MOBILE_DEFAULTS if data or fmt != "tflite_int8"
    ]
    if not formats_to_export:
        raise ValueError("At least one export format must be selected.")

    report = ExportReport(
        source_weights=str(weights_path),
        image_size=imgsz,
        calibration_data=data,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    )

    print("\n" + "=" * 72)
    print("  YOLO MODEL QUANTIZATION & EXPORT FOR MOBILE DEPLOYMENT")
    print("=" * 72)
    print(f"  Source weights : {weights_path}")
    print(f"  Image size     : {imgsz}")
    print(f"  Calibration    : {data or '(none — INT8 omitted unless requested)'}")
    print(f"  Output dir     : {out_dir}")
    print(f"  Formats        : {', '.join(formats_to_export)}")
    print("=" * 72 + "\n")

    for i, fmt in enumerate(formats_to_export, 1):
        print(f"[{i}/{len(formats_to_export)}] Exporting {fmt} ...", end=" ", flush=True)
        try:
            # Reload each time because an export can mutate its model in place.
            model = YOLO(str(weights_path))
            result = _export_one(model, fmt, out_dir, imgsz, data)
        except Exception as exc:
            result = ExportResult(
                format_name=fmt, ultralytics_format="", precision="",
                status="error", error_message=f"Model load or export failed: {exc}",
            )
        report.results.append(result)

        if result.status == "success":
            print(f"✓  {result.file_size_mb:>8.2f} MB  ({result.export_time_s:.1f}s)")
        elif result.status == "skipped":
            print(f"⊘  SKIPPED — {result.error_message}")
        else:
            print(f"✗  ERROR — {result.error_message}")

    report.status = "success" if all(item.status == "success" for item in report.results) else "error"

    # ── Summary table ────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  EXPORT SUMMARY")
    print("=" * 72)
    print(f"  {'Format':<18} {'Precision':<10} {'Size (MB)':>10} {'Time (s)':>10} {'Status':<10}")
    print("  " + "─" * 66)
    for r in report.results:
        size_str = f"{r.file_size_mb:.2f}" if r.status == "success" else "—"
        time_str = f"{r.export_time_s:.1f}" if r.export_time_s > 0 else "—"
        print(f"  {r.format_name:<18} {r.precision:<10} {size_str:>10} {time_str:>10} {r.status:<10}")
    print("=" * 72)

    succeeded = [r for r in report.results if r.status == "success"]
    if succeeded:
        smallest = min(succeeded, key=lambda r: r.file_size_mb)
        print(f"\n  Smallest export : {smallest.format_name} → {smallest.file_size_mb:.2f} MB")
        print("  Select an Android model after validation on the target device.")

    # Save report JSON
    report_path = out_dir / "export_report.json"
    with open(report_path, "w") as f:
        json.dump(asdict(report), f, indent=2)
    print(f"\n  Full report saved to: {report_path}\n")

    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a fine-tuned YOLO model to multiple quantized formats for mobile deployment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--weights", required=True,
        help="Path to the fine-tuned YOLO .pt weights file.",
    )
    parser.add_argument(
        "--out", required=True,
        help="Output directory for all exported models.",
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Input image size for the exported models (default: 640).",
    )
    parser.add_argument(
        "--data",
        help="Path to dataset.yaml for INT8 calibration. Required for tflite_int8 / engine_int8.",
    )
    parser.add_argument(
        "--formats", nargs="+", choices=ALL_FORMATS, default=None,
        help=f"Export formats to generate. Default (mobile): {', '.join(MOBILE_DEFAULTS)} "
             "(INT8 omitted without --data). "
             f"All available: {', '.join(ALL_FORMATS)}.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_exports(
        weights=args.weights,
        out=args.out,
        imgsz=args.imgsz,
        data=args.data,
        formats=args.formats,
    )
    return 0 if report.status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
