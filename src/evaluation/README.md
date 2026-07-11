# RBCCPS Evaluation Presentation Suite

`eval_pres` is the research-facing evaluation and presentation package for the
RBCCPS nighttime streetlight pipeline. It evaluates and visualizes the full
chain:

1. streetlight detection in nighttime phone video,
2. simple track/manifest formation,
3. untrained or trained measurement-block reports,
4. route/audit artifacts for city light-map review.

The current package is intentionally honest about model state. It can run on
raw pretrained detector weights and deterministic/untrained measurement heads,
but metrics that require labels are reported as `not_available` unless matching
ground truth is supplied.

## Setup

From `F:\RBCCPS_Directory\eval_pres`:

```powershell
pip install -r requirements.txt
```

The single requirements file includes plotting, evaluation, video extraction,
YOLO inference, and test dependencies.

Check the environment:

```powershell
python -m eval_pres doctor
```

The doctor command checks Python, OpenCV, FFmpeg, Ultralytics, the default YOLO
model, the bundled sample video, measurement-package imports, and write access.

## Main Commands

Evaluate existing measurement artifacts:

```powershell
python -m eval_pres evaluate `
  --manifest runs/example/clip_manifest.json `
  --reports runs/example/measurement/reports.json `
  --ground-truth path/to/ground_truth.json `
  --model-path models/measurement/pretrained/streetlight_detector_v3/hpc_pull/best.pt `
  --latency-seconds 42.0 `
  --out runs/eval_pres_existing_artifacts
```

Run the bundled video through video extraction, detector/tracking, measurement,
overlay rendering, and evaluation:

```powershell
python -m eval_pres demo-video `
  --video eval_pres/sample_videos/busy_street_20260220_200501_629_1min.mp4 `
  --out runs/eval_pres_busy_street_demo `
  --preset standard
```

Build only the measurement artifact pack:

```powershell
python -m eval_pres artifacts measurement `
  --reports runs/example/measurement/reports.json `
  --manifest runs/example/clip_manifest.json `
  --frame-root runs/example `
  --measurement-dir runs/example/measurement `
  --out runs/example/measurement_artifacts
```

Build only the detection/tracking artifact pack:

```powershell
python -m eval_pres artifacts detection `
  --manifest runs/example/clip_manifest.json `
  --frame-root runs/example `
  --out runs/example/detection_tracking_artifacts
```

Remove generated package caches:

```powershell
python -m eval_pres clean --yes
```

Legacy entrypoints remain available:

```powershell
python -m eval_pres.cli
python -m eval_pres.video_demo
python -m eval_pres.measurement_artifacts
python -m eval_pres.detection_tracking_artifacts
```

## Demo Presets

`demo-video` supports three presets:

- `quick`: lower FPS and fewer measurement tracks for debugging.
- `standard`: practical default for fuller visual demos.
- `full`: more detections/tracks for presentation, with higher memory/runtime
  risk.

Explicit numeric flags such as `--fps-sample`, `--max-det`, and
`--measurement-max-tracks` override preset defaults.

## Inputs

Minimum evaluator inputs:

- `clip_manifest.json`: frame metadata and detector/tracker rows.
- `reports.json`: measurement reports emitted by `rbccps_measurement`.

Optional inputs:

- `ground_truth.json`: frame boxes, physical lamp IDs, statuses, measurement
  labels, attribution labels, and affected-region labels.
- `--route-distance-km`: needed for false positives per kilometer.
- `--latency-seconds`: needed for end-to-end latency reporting.
- `--model-path`: one or more model files/directories for model-size reporting.

Ground truth may be supplied as `lamps[]` or `frames[].lamps[]`. A useful record
looks like:

```json
{
  "physical_lamp_id": "lamp_001",
  "inventory_id": "inv_001",
  "frame_id": 1,
  "bbox_xyxy": [100, 50, 140, 110],
  "status": "on",
  "illumination_class": "adequate",
  "affected_region_polygon": [[80, 160], [220, 160], [220, 260], [80, 260]],
  "served_area_fraction": 0.52,
  "confounder_present": false,
  "target_attribution_correct": true
}
```

## Output Layout

Evaluation:

```text
evaluation_summary.json
metrics.csv
metric_status.csv
plots/
command.json
environment.json
run_summary.json
logs/
```

Detection/tracking artifacts:

```text
detection_summary_table.csv
frame_detection_counts.csv
track_summary.csv
track_summary_table.csv
tracking_events.csv
track_frame_table.csv
track_fragmentation_table.csv
duplicate_track_candidates.csv
detection_tracking_plots/
detections_overlay_frames/
tracking_overlay_frames/
detections_overlay_video.mp4
tracking_overlay_video.mp4
track_cards/
```

Measurement artifacts:

```text
measurement_summary_table.csv
lamp_status_table.csv
affected_region_table.csv
illumination_feature_table.csv
attribution_table.csv
calibration_abstention_table.csv
physical_estimate_table.csv
route_aggregation_table.csv
measurement_flags_table.csv
measurement_plots/
per_lamp_cards/
measurement_contact_sheet.jpg
```

Video demo:

```text
frames/
detections.csv
detections.json
tracks.csv
tracks.json
clip_manifest.json
measurement/
processed_frames/
processed_video.mp4
contact_sheet.jpg
representative_frames/
detection_tracking_artifacts/
measurement_artifacts/
evaluation/
demo_summary.json
command.json
environment.json
run_summary.json
logs/
```

## Metrics

Overall metrics include application detection rate, duplicate physical-lamp
rate, identity switches, fragmentation, inventory match accuracy, false
positives per kilometer, end-to-end latency, and model size.

Detection/tracking metrics include tracked lamp recall, duplicate track rate,
identity switches, AP50, AP50-75, AP50-95, detector precision, detector recall,
center error, and size error.

Measurement metrics include affected public-space region IoU, lamp emission
status macro F1, false target-lamp attribution under confounders,
poor-as-adequate illumination error rate, spatial coverage bias, and temporal
report stability for the same lamp.

Every metric has a direction (`maximize` or `minimize`) and a status
(`computed` or `not_available`) in `metrics.csv` and `metric_status.csv`.

## Bundled Sample Video

The package includes:

```text
sample_videos/busy_street_20260220_200501_629_1min.mp4
sample_videos/busy_street_20260220_200501_629_1min.json
sample_videos/busy_street_20260220_200501_629_1min_preview.jpg
```

The clip is a one-minute busy nighttime road segment with vehicles, storefront
and signage light, pedestrians, parked bikes, and visible streetlights. It is
useful for qualitative demos, but full detector+measurement processing is not a
normal test path because it can be slow and memory-heavy.

## Troubleshooting

- If video extraction fails, install `opencv-python` from `requirements.txt` or
  ensure `ffmpeg` is on `PATH`.
- If YOLO inference fails, run `python -m eval_pres doctor` and verify the model
  path exists.
- If the measurement block runs out of memory, lower `--fps-sample`, lower
  `--max-det`, or lower `--measurement-max-tracks`.
- If many metrics are `not_available`, provide ground truth with boxes,
  physical lamp IDs, status labels, and measurement labels.
- If imports fail with OpenBLAS memory allocation messages, run commands with
  `OPENBLAS_NUM_THREADS=1` in the environment.
- If run folders become cluttered, use `python -m eval_pres clean --yes` for
  package caches and remove old `runs/` outputs manually.

## Validation Policy

For development validation, run unit tests and tiny synthetic smoke tests. Do
not rerun the full bundled 1-minute detector+measurement demo unless a visual
presentation artifact is explicitly needed.
