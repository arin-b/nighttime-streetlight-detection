# RBCCPS Measurement Block Independent Bundle

This bundle contains the measurement-block design documentation, implementation code, tests, model assets, dataset-converter/annotation support code, the exact extracted frame sequence used for the untrained demo, and the final processed measurement demo artifacts.

## What is included

- `documentation/system_design/new_design/`: measurement-block architecture PDF/TeX, module diagrams, annotation/training protocols, and related design notes.
- `src/rbccps_measurement/`: full measurement block implementation, including Modules 1-12, dataset preparation, training hooks, CLI commands, contracts, and reporting.
- `src/rbccps_annotator/`: annotation/review workspace code linked to measurement-label preparation.
- `scripts/measurement/`: final sweep runner, video demo runner, and measurement smoke script.
- `scripts/annotation_automation/`, `scripts/annotator_bundle/`, and root annotation import/export scripts: tooling linked to collecting/reviewing data for the measurement block.
- `tests/`: current test suite, including measurement module tests, converter tests, pipeline tests, and video-demo tests.
- `models/measurement/`: pretrained/runtime assets currently used by the measurement demo and model registry, including the streetlight detector weights.
- `datasets/extracted_frames/mobile_night_videos/2025-05-29/20250529_2050207/`: the exact 100-frame source sequence used for the 33-second untrained demo.
- `runs/measurement_final_sweep_20260705_233927/`: passed final sweep output, processed video, contact sheet, clip manifest, reports, logs, and summary JSON.
- `exports/annotations_LLM/`, selected ChatGPT/annotation manifests, and root annotation JSONs: current annotation examples/workflow evidence linked to the annotation-to-measurement converter.

## Key demo artifacts

- `runs/measurement_final_sweep_20260705_233927/final_sweep_summary.json`
- `runs/measurement_final_sweep_20260705_233927/video_demo/processed_demo.mp4`
- `runs/measurement_final_sweep_20260705_233927/video_demo/contact_sheet.png`
- `runs/measurement_final_sweep_20260705_233927/video_demo/clip_manifest.json`
- `runs/measurement_final_sweep_20260705_233927/video_demo/measurement/reports.json`

## Re-run the demo

From the unzipped bundle root, install the project in an environment with the measurement extras, then run:

```powershell
python -m pip install -e .[measurement,dev]
python scripts/measurement/run_measurement_video_demo.py `
  --frames-dir datasets/extracted_frames/mobile_night_videos/2025-05-29/20250529_2050207 `
  --out runs/measurement_demo_rerun `
  --fps 3 `
  --max-frames 100 `
  --conf 0.05 `
  --batch-size 4
```

Run the final sweep:

```powershell
python scripts/measurement/run_final_sweep.py --fps 3 --max-frames 100 --conf 0.05 --batch-size 4
```

## Notes on intentional exclusions

The full raw mobile-video corpus under `datasets/raw/mobile_night_videos/` is about 9.7 GB and is not required to run the included measurement-block demo. This bundle includes the exact extracted 100-frame sequence used for the demo instead. Redundant/generated archive-only exports such as older bulk ChatGPT zip batches and portable annotator bundle zips are not required for the measurement block runtime; current annotation examples, manifests, scripts, and converter code are included.

All included demo outputs are untrained deterministic measurement-block outputs, not calibrated/trained performance claims.
