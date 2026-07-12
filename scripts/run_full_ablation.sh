#!/usr/bin/env bash
# run_full_ablation.sh
# 
# All-in-one script to run training for ablation cases and evaluate them through
# the end-to-end streetlight audit pipeline.

set -e

# Configuration variables
VIDEO_PATH="${VIDEO_PATH:-}"
GT_LABELS="${GT_LABELS:-}"
EXPERIMENT="${EXPERIMENT:-original}"
CASES=("baseline" "geometry" "cse" "geometry_cse" "negative" "negative_cse" "negative_geometry" "all_modules")

usage() {
    echo "Usage: $0 --video <video_path> --gt-labels <labels_dir> [options]"
    echo ""
    echo "Options:"
    echo "  --video        Path to the test video for evaluation."
    echo "  --gt-labels    Path to the YOLO-format ground truth labels directory."
    echo "  --experiment   Dataset experiment name (default: 'original'). Options: original, zerodce, retinex."
    echo "  --cases        Comma-separated list of cases to run, or 'all'. (default: all)"
    exit 1
}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --video) VIDEO_PATH="$2"; shift ;;
        --gt-labels) GT_LABELS="$2"; shift ;;
        --experiment) EXPERIMENT="$2"; shift ;;
        --cases) IFS=',' read -ra CASES <<< "$2"; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

if [[ -z "$VIDEO_PATH" || -z "$GT_LABELS" ]]; then
    echo "Error: --video and --gt-labels are required for the evaluation phase."
    usage
fi

if [[ "${CASES[0]}" == "all" ]]; then
    CASES=("baseline" "geometry" "cse" "geometry_cse" "negative" "negative_cse" "negative_geometry" "all_modules")
fi

# Ensure PYTHONPATH includes the src directory
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

echo "========================================================================"
echo " Starting Full Ablation Study"
echo " Experiment: $EXPERIMENT"
echo " Cases: ${CASES[*]}"
echo " Evaluation Video: $VIDEO_PATH"
echo " Ground Truth: $GT_LABELS"
echo "========================================================================"

# Step 1: Training Phase
for CASE in "${CASES[@]}"; do
    echo ""
    echo ">>> Phase 1: Training $CASE for experiment $EXPERIMENT"
    echo "------------------------------------------------------------------------"
    python3 -m src.detection.training.ablation \
        --experiment "$EXPERIMENT" \
        --case "$CASE"
done

# Step 2: Audit & Evaluation Phase
for CASE in "${CASES[@]}"; do
    echo ""
    echo ">>> Phase 2: Evaluating $CASE through the full audit pipeline"
    echo "------------------------------------------------------------------------"
    
    # Resolve the model path based on naming convention in ablation.py
    # E.g., for experiment 'original' and case 'baseline', the output run name is usually 'streetlight_yolo26m_original_baseline'
    # Wait, the artifact labels use hyphens (e.g. geometry-cse). The run label uses underscores.
    # Let's dynamically find the latest weights for this run pattern.
    
    # We will search for runs matching the expected name
    MODEL_WEIGHTS=$(find runs/train -maxdepth 3 -type f -path "*/streetlight_yolo26m_${EXPERIMENT}_${CASE}*/weights/best.pt" | sort -r | head -n 1)
    
    if [[ -z "$MODEL_WEIGHTS" ]]; then
        echo "[WARNING] Could not find trained weights for $CASE in runs/train/streetlight_yolo26m_${EXPERIMENT}_${CASE}*/weights/best.pt"
        echo "[WARNING] Skipping evaluation for $CASE."
        continue
    fi
    
    OUTPUT_DIR="runs/audit/ablation_${EXPERIMENT}_${CASE}"
    
    echo "Using weights: $MODEL_WEIGHTS"
    echo "Output dir:    $OUTPUT_DIR"
    
    python3 -m src.evaluation.eval_pres.run_audit \
        --video "$VIDEO_PATH" \
        --model "$MODEL_WEIGHTS" \
        --gt-labels "$GT_LABELS" \
        --output-dir "$OUTPUT_DIR"
        
    echo "[OK] Evaluation for $CASE completed. Results saved in $OUTPUT_DIR"
done

echo ""
echo "========================================================================"
echo " Ablation Study Completed Successfully!"
echo " Check runs/audit/ for the comprehensive JSON/CSV/MD reports."
echo "========================================================================"
