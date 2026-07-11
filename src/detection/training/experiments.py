from __future__ import annotations

import json
from pathlib import Path


def summarize_experiment_layout(dataset_root: str | Path, run_name: str, notes: str = "") -> dict[str, str]:
    dataset_root = str(Path(dataset_root))
    return {"dataset_root": dataset_root, "run_name": run_name, "notes": notes}


def main() -> None:
    print(json.dumps({"status": "placeholder", "message": "Use train/validate plus dataset builders to compose experiments."}, indent=2))


if __name__ == "__main__":
    main()
