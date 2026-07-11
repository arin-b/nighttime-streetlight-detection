from __future__ import annotations

import sys

from rbccps_annotator.cli import main as annotator_main


def main() -> None:
    print(
        "The old streetlight review app has been replaced by the modular RBCCPS annotator.\n"
        "Use one of:\n"
        "  python -m rbccps_annotator ingest-frames --frames <frame_dir> --workspace <workspace>\n"
        "  python -m rbccps_annotator ingest-detector-run --manifest <clip_manifest.json> --workspace <workspace>\n"
        "  python -m rbccps_annotator serve --workspace <workspace>\n"
        "\n"
        "Compatibility: forwarding arguments after review-app to rbccps_annotator.\n"
    )
    sys.argv = ["python -m rbccps_annotator", *sys.argv[1:]]
    if len(sys.argv) == 1:
        sys.argv.append("--help")
    annotator_main()


if __name__ == "__main__":
    main()
