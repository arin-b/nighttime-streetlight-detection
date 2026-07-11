from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from .artifact_utils import repo_root


def clean_eval_pres(*, yes: bool = False) -> dict[str, Any]:
    root = repo_root() / "eval_pres"
    targets = sorted(
        [path for path in root.rglob("__pycache__") if path.is_dir()]
        + [path for path in root.rglob(".pytest_cache") if path.is_dir()]
    )
    removed: list[str] = []
    for target in targets:
        if yes:
            shutil.rmtree(target, ignore_errors=True)
            removed.append(str(target))
    return {
        "root": str(root),
        "mode": "deleted" if yes else "dry_run",
        "targets": [str(path) for path in targets],
        "removed": removed,
        "note": "Run with --yes to delete listed cache directories." if not yes else "Cache directories removed.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remove generated eval_pres cache directories.")
    parser.add_argument("--yes", action="store_true", help="Actually delete cache directories. Without this flag, only lists targets.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(clean_eval_pres(yes=args.yes), indent=2))


if __name__ == "__main__":
    main()
