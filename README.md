# Nighttime Streetlight Detection & Audit Pipeline

A production-ready, modular computer vision pipeline designed for nighttime streetlight detection, tracking, measurement, and automated auditing. 

This project integrates advanced YOLOv26 detection with attention modules (Geometry, Channel Squeeze-Excitation, Negative Attention), multi-object tracking (BoT-SORT & UCMC), robust multi-cue temporal filtering, and photometric measurement to generate comprehensive infrastructure audit reports.

---

## 🚀 Features

- **Advanced Detection Models**: YOLOv26 baseline with dynamically configurable attention modules (CSE, Geometry, Negative Masking).
- **Ablation Framework**: Built-in W&B logging and orchestration for running complete module ablation studies.
- **Robust Tracking**: Integration with BoT-SORT (and upcoming UCMC) for stable streetlight identity maintenance across frames.
- **Multi-Cue Filtering**: Eradicates phantom detections using spatial priors, aspect-ratio consistency, brightness, and temporal stability checks.
- **Photometric Measurement**: Real-time evaluation of streetlight brightness, status (Working, Off, Flickering), and illumination profiles.
- **Location Prior**: GPS-aware memory system to correlate detected streetlights with known historical infrastructure data.
- **Comprehensive Auditing**: End-to-end evaluation producing JSON, CSV, and Markdown audit reports with built-in mAP and F1 classification metrics.

---

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.10+
- PyTorch 2.3.0+
- A CUDA-compatible GPU (recommended for real-time inference)

### Installation

1. **Clone the repository and enter the directory**:
   ```bash
   git clone https://github.com/your-username/nighttime_streetlight_detection.git
   cd nighttime_streetlight_detection
   ```

2. **Install the package and dependencies**:
   The project uses `pyproject.toml`. Install the core package and all optional dependencies (`train`, `measurement`, `annotator`, `dev`) in editable mode:
   ```bash
   pip install -e ".[train,measurement,annotator,dev]"
   ```

3. **Verify installation**:
   ```bash
   python3 -c "import rbccps_od; print('Success!')"
   ```

### Installation via Docker (Containerization)

For the most reliable and consistent environment, you can build and run this repository inside a Docker container. The container comes pre-configured with PyTorch, CUDA, and OpenCV dependencies.

1. **Build the Docker Image**:
   ```bash
   docker build -t streetlight-audit .
   ```

2. **Run the Container**:
   You should mount your local `datasets`, `runs`, and weights directories so the container can access your data and save outputs persistently.
   ```bash
   docker run --gpus all -it --rm \
       -v $(pwd)/runs:/app/runs \
       -v $(pwd)/datasets:/app/datasets \
       -v /path/to/your/weights:/app/weights \
       streetlight-audit
   ```
   *Inside the container, you can then run any of the commands listed below!*

---

## 🧪 Training & Ablation Studies

The project includes an advanced training orchestrator to run ablation studies on the detection attention modules. 

### Running a Specific Ablation Case
You can train a specific model variant using `ablation.py`.

```bash
python3 -m src.detection.training.ablation --experiment original --case baseline
```

**Available Experiments**:
- `original` (Baseline original images)
- `zerodce` (Zero-DCE enhanced images)
- `retinex` (Retinex decomposition)

**Available Cases**:
- `baseline` (Standard YOLO)
- `geometry` (Geometry-aware attention only)
- `cse` (Channel Squeeze-Excitation only)
- `geometry_cse` (Geometry + CSE)
- `negative` (Negative Attention Mask only)
- `negative_cse` (Negative + CSE)
- `negative_geometry` (Negative + Geometry)
- `all_modules` (All three attention modules active)

### 🚀 All-In-One Ablation Script
To run an end-to-end ablation study (training all 8 module configurations and subsequently evaluating them through the audit pipeline), use the provided bash script:

```bash
./scripts/run_full_ablation.sh \
    --experiment original \
    --video /path/to/test_video.mp4 \
    --gt-labels /path/to/yolo_labels_dir/ \
    --cases all
```

---

## 🔦 Running the Audit Pipeline

The core evaluation architecture integrates detection, tracking, measurement, and aggregation into a single command. The orchestrator produces a marked-up video alongside comprehensive `.csv`, `.json`, and `.md` reports.

### Basic Execution
```bash
python3 -m src.evaluation.eval_pres.run_audit \
    --video /path/to/video.mp4 \
    --model /path/to/best_weights.pt \
    --output-dir runs/audit/my_audit_run
```

### Advanced Execution with Full Evaluation
To evaluate detection accuracy (mAP, F1) and status classification alongside the audit, provide ground-truth YOLO labels and a status CSV:

```bash
python3 -m src.evaluation.eval_pres.run_audit \
    --video /path/to/video.mp4 \
    --model /path/to/best_weights.pt \
    --gt-labels /path/to/yolo_labels/ \
    --gt-status-file /path/to/ground_truth_status.csv \
    --output-dir runs/audit/full_eval_run
```

### Using the Location Prior
If you have telemetry data and want to correlate detections with known GPS locations:

```bash
python3 -m src.evaluation.eval_pres.run_audit \
    --video /path/to/video.mp4 \
    --model /path/to/best_weights.pt \
    --location-prior runs/audit/known_lamp_prior.json \
    --location-samples /path/to/telemetry.csv
```

---

## 📜 Complete List of Commands

Here is a quick reference guide to the most common tasks:

| Task | Command |
|------|---------|
| **Install Package** | `pip install -e ".[train,measurement,dev]"` |
| **Train Baseline** | `python3 -m src.detection.training.ablation --experiment original --case baseline` |
| **Train All Modules** | `python3 -m src.detection.training.ablation --experiment original --case all_modules` |
| **Full Ablation Run** | `./scripts/run_full_ablation.sh --experiment original --video <video.mp4> --gt-labels <labels_dir>` |
| **Run Basic Audit** | `python3 -m src.evaluation.eval_pres.run_audit --video <video.mp4> --model <best.pt>` |
| **Audit with Eval** | `python3 -m src.evaluation.eval_pres.run_audit --video <vid> --model <pt> --gt-labels <dir>` |
| **Audit Dry-Run** | `python3 -m src.evaluation.eval_pres.run_audit --video <vid> --model <pt> --dry-run` |

---

## 🏗️ Architecture Map

Following a recent architectural merger, the core systems reside in:
- `src/detection/models/`: Core YOLO adaptors and attention definitions.
- `src/detection/training/`: Training and ablation frameworks.
- `src/detection/pipeline/`: Trackers and multicue filters.
- `src/evaluation/eval_pres/`: The main orchestrator (`run_audit.py`), measurement engines, metrics, and report generators.

---
*Developed by the RBCCPS Computer Vision Team.*
