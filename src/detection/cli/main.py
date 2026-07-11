from __future__ import annotations

import importlib
import sys

COMMAND_TO_MODULE = {
    "review-app": "rbccps_od.review.app",
    "annotator": "rbccps_annotator.cli",
    "sync-v3-reviews": "rbccps_od.datasets.review_sync",
    "build-v3-corpus": "rbccps_od.datasets.corpus_v3",
    "train": "rbccps_od.training.train",
    "train-original": "rbccps_od.training.train_original",
    "train-ablation": "rbccps_od.training.ablation",
    "validate": "rbccps_od.training.validate",
    "build-tiled": "rbccps_od.datasets.tiled",
    "build-mixed": "rbccps_od.datasets.mixed_external",
    "download-models": "rbccps_od.models.downloader",
    "run-baseline": "rbccps_od.pipeline.baseline_detector",
    "run-advanced-pipeline": "rbccps_od.pipeline.advanced_runner",
    "run-video-pipeline": "rbccps_od.pipeline.video_runner",
    "export-candidates": "rbccps_od.pipeline.export_candidates",
    "score-reliability": "rbccps_od.evaluation.reliability",
    "evaluate-gate": "rbccps_od.evaluation.gate",
    "integrate-reviewed-data": "rbccps_od.datasets.negative_integration",
    "materialize-review-batches": "rbccps_od.review.materialize_batches",
    "build-review-subset": "rbccps_od.review.subset",
    "propagate-reviews": "rbccps_od.review.propagation",
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        commands = "\n".join(f"  - {name}" for name in COMMAND_TO_MODULE)
        print(f"Usage: python -m rbccps_od <command> [args...]\nAvailable commands:\n{commands}")
        return
    command = sys.argv[1]
    module_name = COMMAND_TO_MODULE.get(command)
    if not module_name:
        raise SystemExit(f"Unknown command: {command}")
    module = importlib.import_module(module_name)
    sys.argv = [f"{sys.argv[0]} {command}", *sys.argv[2:]]
    module.main()
