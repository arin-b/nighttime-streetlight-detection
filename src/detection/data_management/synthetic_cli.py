from __future__ import annotations

import argparse
import json
from pathlib import Path

from rbccps_od.data_management.synthetic import generate, load_config, statistics_from_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate auditable, illumination-only streetlight training variants.")
    subcommands = parser.add_subparsers(dest="command", required=True)
    generate_parser = subcommands.add_parser("generate", help="select sources and generate a separate synthetic training pool")
    generate_parser.add_argument("--config", required=True)
    generate_parser.add_argument("--dry-run", action="store_true", help="report source selection without writing images, labels, or metadata")
    generate_parser.add_argument("--limit", type=int, help="maximum synthetic images to write (recommended for pilot review)")
    generate_parser.add_argument("--seed", type=int, help="override random_seed in the config")
    generate_parser.add_argument("--preview", action="store_true", help="write original/mask/synthetic preview sheets")
    stats_parser = subcommands.add_parser("stats", help="summarise an existing JSONL metadata file")
    stats_parser.add_argument("--metadata", required=True)
    args = parser.parse_args()
    if args.command == "stats":
        print(json.dumps(statistics_from_metadata(Path(args.metadata)), indent=2, sort_keys=True))
        return
    config = load_config(args.config)
    if args.dry_run:
        config.dry_run = True
    if args.seed is not None:
        config.random_seed = args.seed
    if args.preview:
        config.generate_preview = True
    report = generate(config, limit=args.limit)
    print(json.dumps(report, indent=2, sort_keys=True))
    for note in report.get("warnings", []):
        print(f"WARNING: {note}")


if __name__ == "__main__":
    main()
