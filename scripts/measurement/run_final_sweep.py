from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_FRAMES_DIR = Path("datasets/extracted_frames/mobile_night_videos/2025-05-29/20250529_2050207")
DEFAULT_SELECTED_SAMPLES = Path("runs/measurement_pseudo_annotation_classes/selected_samples.json")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def run_step(
    name: str,
    command: list[str],
    log_path: Path,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as handle:
        handle.write("$ " + " ".join(command) + "\n\n")
        handle.flush()
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return {
        "name": name,
        "status": "passed" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "log": str(log_path),
    }


def run_final_sweep(
    out_root: Path,
    frames_dir: Path,
    fps: float,
    max_frames: int,
    conf: float,
    batch_size: int,
    skip_pytest: bool = False,
    skip_smoke: bool = False,
    skip_demo: bool = False,
) -> dict[str, Any]:
    repo = repo_root()
    out_root = out_root if out_root.is_absolute() else repo / out_root
    frames_dir = frames_dir if frames_dir.is_absolute() else repo / frames_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_root / f"measurement_final_sweep_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = run_dir / "tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["TEMP"] = str(temp_dir)
    env["TMP"] = str(temp_dir)

    steps: list[dict[str, Any]] = []
    compile_targets = [
        "src/rbccps_measurement",
        "scripts/measurement/run_measurement_video_demo.py",
        "scripts/measurement/run_final_sweep.py",
    ]
    steps.append(
        run_step(
            "compile",
            [sys.executable, "-m", "compileall", "-q", *compile_targets],
            run_dir / "compile.log",
            cwd=repo,
            env=env,
        )
    )

    if skip_pytest:
        steps.append({"name": "pytest", "status": "skipped", "reason": "requested by --skip-pytest"})
    else:
        steps.append(
            run_step(
                "pytest",
                [sys.executable, "-m", "pytest", "-p", "no:cacheprovider"],
                run_dir / "pytest.log",
                cwd=repo,
                env=env,
            )
        )

    selected_samples = repo / DEFAULT_SELECTED_SAMPLES
    if skip_smoke:
        steps.append({"name": "all_slots_smoke", "status": "skipped", "reason": "requested by --skip-smoke"})
    elif not selected_samples.exists():
        steps.append(
            {
                "name": "all_slots_smoke",
                "status": "skipped",
                "reason": f"missing {_relative(selected_samples, repo)}",
            }
        )
    else:
        steps.append(
            run_step(
                "all_slots_smoke",
                [sys.executable, "scripts/measurement/run_all_slots_smoke.py"],
                run_dir / "all_slots_smoke.log",
                cwd=repo,
                env=env,
            )
        )

    video_demo_dir = run_dir / "video_demo"
    if skip_demo:
        steps.append({"name": "video_demo", "status": "skipped", "reason": "requested by --skip-demo"})
    else:
        steps.append(
            run_step(
                "video_demo",
                [
                    sys.executable,
                    "scripts/measurement/run_measurement_video_demo.py",
                    "--frames-dir",
                    str(frames_dir),
                    "--out",
                    str(video_demo_dir),
                    "--fps",
                    str(fps),
                    "--max-frames",
                    str(max_frames),
                    "--conf",
                    str(conf),
                    "--batch-size",
                    str(batch_size),
                ],
                run_dir / "video_demo.log",
                cwd=repo,
                env=env,
            )
        )

    hard_failures = [
        step
        for step in steps
        if step["status"] == "failed" and step["name"] in {"compile", "pytest", "all_slots_smoke", "video_demo"}
    ]
    summary = {
        "implementation": "measurement_final_sweep_v1",
        "run_dir": str(run_dir),
        "status": "failed" if hard_failures else "passed",
        "timestamp": timestamp,
        "steps": steps,
        "artifacts": {
            "pytest_log": str(run_dir / "pytest.log"),
            "video_demo_dir": str(video_demo_dir),
            "processed_demo": str(video_demo_dir / "processed_demo.mp4"),
            "contact_sheet": str(video_demo_dir / "contact_sheet.png"),
            "clip_manifest": str(video_demo_dir / "clip_manifest.json"),
            "measurement_reports": str(video_demo_dir / "measurement" / "reports.json"),
            "demo_summary": str(video_demo_dir / "demo_summary.json"),
        },
        "notes": [
            "Demo outputs are untrained deterministic measurement-block outputs.",
            "The all-slots smoke script is skipped when its selected sample manifest is absent.",
        ],
    }
    (run_dir / "final_sweep_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run final RBCCPS measurement verification and untrained video demo.")
    parser.add_argument("--out-root", default="runs", help="Root directory for the timestamped sweep output.")
    parser.add_argument("--frames-dir", default=str(DEFAULT_FRAMES_DIR), help="Extracted frame sequence for the demo.")
    parser.add_argument("--fps", type=float, default=3.0, help="Rendered demo FPS.")
    parser.add_argument("--max-frames", type=int, default=100, help="Maximum extracted frames to process.")
    parser.add_argument("--conf", type=float, default=0.05, help="YOLO confidence threshold for the demo.")
    parser.add_argument("--batch-size", type=int, default=4, help="YOLO batch size for the demo.")
    parser.add_argument("--skip-pytest", action="store_true", help="Skip full pytest.")
    parser.add_argument("--skip-smoke", action="store_true", help="Skip all-slots smoke script.")
    parser.add_argument("--skip-demo", action="store_true", help="Skip the video demo.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_final_sweep(
        out_root=Path(args.out_root),
        frames_dir=Path(args.frames_dir),
        fps=args.fps,
        max_frames=args.max_frames,
        conf=args.conf,
        batch_size=args.batch_size,
        skip_pytest=args.skip_pytest,
        skip_smoke=args.skip_smoke,
        skip_demo=args.skip_demo,
    )
    print(json.dumps(summary, indent=2))
    if summary["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
