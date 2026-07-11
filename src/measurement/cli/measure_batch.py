from __future__ import annotations

import argparse
import json
from pathlib import Path

from rbccps_measurement.pipeline import run_clip_to_directory
from rbccps_measurement.route.graph_aggregation import aggregate_route_reports, write_route_aggregation_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run measurement-block inference for every clip in a prepared dataset.")
    parser.add_argument("--dataset", required=True, help="Prepared dataset root containing dataset_manifest.json.")
    parser.add_argument("--out", required=True, help="Output batch directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = Path(args.dataset)
    payload = json.loads((dataset / "dataset_manifest.json").read_text(encoding="utf-8-sig"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    completed = []
    all_reports = []
    source_report_refs = {}
    for item in payload.get("clips", []):
        manifest_ref = Path(item["manifest"])
        manifest_path = manifest_ref if manifest_ref.is_absolute() else dataset / manifest_ref
        clip_out = out / item["clip_id"]
        reports = run_clip_to_directory(manifest_path, clip_out)
        all_reports.extend(reports)
        for report in reports:
            source_report_refs[report.lamp_observation_id] = str(clip_out / "reports.json")
        completed.append({"clip_id": item["clip_id"], "reports": len(reports), "out": str(clip_out)})
    route_group = str(payload.get("route_group") or payload.get("route_id") or "unknown_route")
    aggregation = aggregate_route_reports(all_reports, source_report_refs=source_report_refs, route_group=route_group)
    write_route_aggregation_outputs(aggregation, out)
    (out / "batch_summary.json").write_text(
        json.dumps(
            {
                "clips": completed,
                "route_aggregation": {
                    "candidate_lamps": len(aggregation.lamps),
                    "road_segments": len(aggregation.road_segments),
                    "audit_trail": "audit_trail.json",
                    "route_aggregation_ref": "route_aggregation.json",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"out": str(out), "clips": len(completed)}, indent=2))


if __name__ == "__main__":
    main()
