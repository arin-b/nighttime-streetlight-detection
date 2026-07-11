from __future__ import annotations

import importlib
import sys

COMMAND_TO_MODULE = {
    "check-readiness": "rbccps_measurement.cli.check_readiness",
    "download-models": "rbccps_measurement.models.downloader",
    "pseudo-manifest": "rbccps_measurement.cli.pseudo_manifest",
    "prepare-annotations": "rbccps_measurement.cli.prepare_annotations",
    "prepare-dataset": "rbccps_measurement.cli.prepare_dataset",
    "measure-clip": "rbccps_measurement.cli.measure_clip",
    "measure-batch": "rbccps_measurement.cli.measure_batch",
    "evaluate": "rbccps_measurement.cli.evaluate",
    "train-module": "rbccps_measurement.cli.train_module",
    "train-joint": "rbccps_measurement.cli.train_joint",
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        commands = "\n".join(f"  - {name}" for name in COMMAND_TO_MODULE)
        print(f"Usage: python -m rbccps_measurement <command> [args...]\nAvailable commands:\n{commands}")
        return
    command = sys.argv[1]
    module_name = COMMAND_TO_MODULE.get(command)
    if not module_name:
        raise SystemExit(f"Unknown command: {command}")
    module = importlib.import_module(module_name)
    sys.argv = [f"{sys.argv[0]} {command}", *sys.argv[2:]]
    module.main()
