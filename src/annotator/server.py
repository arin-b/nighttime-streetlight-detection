from __future__ import annotations

import json
import mimetypes
import threading
import urllib.parse
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Any

from rbccps_annotator.auto_polygon import prompt_segmenter_status, propose_auto_polygon
from rbccps_annotator.exports import export_measurement, export_yolo
from rbccps_annotator.schema import (
    ATTRIBUTION_CLASSES,
    LAMP_STATUS_CLASSES,
    LAMP_BOX_CLASSES,
    LUX_POINT_TYPES,
    PUBLIC_SPACE_TYPES,
    SURFACE_TYPES,
    VISIBILITY_CLASSES,
)
from rbccps_annotator.workspace import item_lookup, load_manifest, load_review, read_json, save_review, write_json


class AnnotatorServer:
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace.resolve()
        self.items = list(load_manifest(self.workspace).get("items", []))
        if not self.items:
            raise ValueError(f"Workspace has no items: {self.workspace}")
        self.item_by_key = {item["key"]: item for item in self.items}
        self._export_timer: threading.Timer | None = None
        self._export_lock = threading.Lock()

    def bootstrap(self) -> dict[str, Any]:
        reviewed = 0
        for item in self.items:
            review = load_review(self.workspace, item)
            if review.get("review_status") and review.get("review_status") != "unreviewed":
                reviewed += 1
        return {
            "workspace": str(self.workspace),
            "total": len(self.items),
            "reviewed": reviewed,
            "modes": [
                "od_standard",
                "od_confounder",
                "track_review",
                "lamp_status",
                "public_space",
                "affected_region",
                "task_visibility",
                "attribution",
                "lux_reference",
                "qa",
            ],
            "surface_types": SURFACE_TYPES,
            "lamp_box_classes": LAMP_BOX_CLASSES,
            "public_space_types": PUBLIC_SPACE_TYPES,
            "lamp_status_classes": LAMP_STATUS_CLASSES,
            "visibility_classes": VISIBILITY_CLASSES,
            "attribution_classes": ATTRIBUTION_CLASSES,
            "lux_point_types": LUX_POINT_TYPES,
            "auto_polygon": prompt_segmenter_status(),
            "bundle_state": self.bundle_state(),
            "tutorial": self.tutorial_manifest(),
        }

    def get_item(self, key: str | None = None, index: int | None = None) -> dict[str, Any]:
        if key:
            item = self.item_by_key.get(key)
            if not item:
                raise ValueError(f"Unknown key: {key}")
        else:
            item = self.items[max(0, min(index or 0, len(self.items) - 1))]
        item_index = self.items.index(item)
        return {
            "item": item,
            "review": load_review(self.workspace, item),
            "index": item_index,
            "total": len(self.items),
            "prev_key": self.items[item_index - 1]["key"] if item_index > 0 else None,
            "next_key": self.items[item_index + 1]["key"] if item_index + 1 < len(self.items) else None,
        }

    def image_path(self, key: str) -> Path:
        current = item_lookup(self.workspace).get(key)
        if not current:
            raise ValueError(f"Unknown key: {key}")
        path = Path(current["image_path"])
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    def auto_polygon(self, payload: dict[str, Any]) -> dict[str, Any]:
        key = str(payload["item_key"])
        image_path = self.image_path(key)
        protected_boxes = payload.get("protected_boxes")
        if protected_boxes is None:
            item = self.item_by_key.get(key)
            protected_boxes = []
            if item:
                review = load_review(self.workspace, item)
                protected_boxes = [box.get("bbox_xyxy", []) for box in review.get("boxes", [])]
        return propose_auto_polygon(
            image_path=image_path,
            bbox_xyxy=[float(value) for value in payload["bbox_xyxy"]],
            protected_boxes=protected_boxes,
            margin_px=int(payload.get("margin_px", 12)),
        ).to_dict()

    def bundle_state(self) -> dict[str, Any]:
        return read_json(self.workspace / "bundle_state.json", {})

    def tutorial_manifest(self) -> dict[str, Any]:
        manifest = read_json(self.workspace / "tutorial_manifest.json", {"examples": [], "warnings": []})
        examples = []
        for index, example in enumerate(manifest.get("examples", [])):
            examples.append(
                {
                    "id": example.get("id", f"tutorial_{index + 1}"),
                    "title": example.get("title", f"Tutorial {index + 1}"),
                    "lesson": example.get("lesson", ""),
                    "index": index,
                }
            )
        return {"examples": examples, "warnings": manifest.get("warnings", [])}

    def get_tutorial_item(self, index: int = 0) -> dict[str, Any]:
        manifest = read_json(self.workspace / "tutorial_manifest.json", {"examples": []})
        examples = manifest.get("examples", [])
        if not examples:
            raise ValueError("No tutorial examples are available.")
        index = max(0, min(index, len(examples) - 1))
        example = examples[index]
        image_path = Path(example["image_path"])
        with image_path.open("rb"):
            pass
        item = {
            "key": f"tutorial_{example.get('id', index + 1)}",
            "image_id": example.get("id", f"tutorial_{index + 1}"),
            "image_path": str(image_path),
            "width": 0,
            "height": 0,
            "clip_id": "tutorial",
            "frame_id": str(index + 1),
            "split": "tutorial",
            "source_pool": "tutorial_gold",
            "metadata": {"tutorial": True},
        }
        try:
            from PIL import Image

            with Image.open(image_path) as image:
                item["width"], item["height"] = image.size
        except Exception:
            pass
        return {
            "item": item,
            "review": example.get("review", {}),
            "gold_review": example.get("review", {}),
            "index": index,
            "total": len(examples),
            "prev_key": str(index - 1) if index > 0 else None,
            "next_key": str(index + 1) if index + 1 < len(examples) else None,
            "tutorial": {
                "id": example.get("id", f"tutorial_{index + 1}"),
                "title": example.get("title", f"Tutorial {index + 1}"),
                "lesson": example.get("lesson", ""),
            },
        }

    def tutorial_image_path(self, key: str) -> Path:
        manifest = read_json(self.workspace / "tutorial_manifest.json", {"examples": []})
        key = key.removeprefix("tutorial_")
        for index, example in enumerate(manifest.get("examples", [])):
            if key in {str(index), str(index + 1), str(example.get("id", ""))}:
                path = Path(example["image_path"])
                if not path.exists():
                    raise FileNotFoundError(path)
                return path
        raise ValueError(f"Unknown tutorial key: {key}")

    def mark_tutorial_completed(self, completed: bool = True) -> dict[str, Any]:
        state_path = self.workspace / "bundle_state.json"
        state = read_json(state_path, {})
        state["tutorial_completed"] = completed
        state["tutorial_completed_at"] = time_now()
        write_json(state_path, state)
        return state

    def save_review_and_export(self, key: str, review: dict[str, Any]) -> dict[str, Any]:
        saved = save_review(self.workspace, key, review)
        self.schedule_auto_export()
        return saved

    def export_status(self) -> dict[str, Any]:
        state = self.bundle_state()
        export_root = Path(state.get("exports") or self.workspace / "exports" / "auto")
        return read_json(
            export_root / "export_summary.json",
            {"workspace": str(self.workspace), "updated_at": "", "status": "not_started"},
        )

    def schedule_auto_export(self) -> None:
        state = self.bundle_state()
        export_root = Path(state.get("exports") or self.workspace / "exports" / "auto")
        export_root.mkdir(parents=True, exist_ok=True)
        write_json(export_root / "export_summary.json", {"workspace": str(self.workspace), "updated_at": time_now(), "status": "queued"})

        def run_export() -> None:
            with self._export_lock:
                try:
                    write_json(export_root / "export_summary.json", {"workspace": str(self.workspace), "updated_at": time_now(), "status": "running"})
                    export_yolo(self.workspace, export_root / "yolo", split_dirs=True)
                    export_measurement(self.workspace, export_root / "measurement")
                    write_json(export_root / "export_summary.json", {"workspace": str(self.workspace), "updated_at": time_now(), "status": "ok"})
                except Exception as error:
                    write_json(export_root / "export_summary.json", {"workspace": str(self.workspace), "updated_at": time_now(), "status": "error", "error": str(error)})

        if self._export_timer:
            self._export_timer.cancel()
        self._export_timer = threading.Timer(1.0, run_export)
        self._export_timer.daemon = True
        self._export_timer.start()


def time_now() -> str:
    import time

    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def run_server(workspace: Path, host: str, port: int, open_browser: bool) -> None:
    app = AnnotatorServer(workspace)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            return

        def _send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_error_json(self, error: Exception, status: HTTPStatus = HTTPStatus.BAD_REQUEST) -> None:
            self._send_json({"error": str(error)}, status=status)

        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            params = urllib.parse.parse_qs(parsed.query)
            try:
                if parsed.path == "/":
                    return self._send_static("index.html")
                if parsed.path.startswith("/static/"):
                    return self._send_static(parsed.path.removeprefix("/static/"))
                if parsed.path == "/api/bootstrap":
                    return self._send_json(app.bootstrap())
                if parsed.path == "/api/export-status":
                    return self._send_json(app.export_status())
                if parsed.path == "/api/item":
                    key = params.get("key", [None])[0]
                    index_text = params.get("index", [None])[0]
                    index = int(index_text) if index_text is not None else None
                    return self._send_json(app.get_item(key=key, index=index))
                if parsed.path == "/api/tutorial/item":
                    index_text = params.get("index", ["0"])[0]
                    return self._send_json(app.get_tutorial_item(index=int(index_text)))
                if parsed.path == "/image":
                    key = params.get("key", [""])[0]
                    if key.startswith("tutorial_") and key not in app.item_by_key:
                        return self._send_file(app.tutorial_image_path(key))
                    return self._send_file(app.image_path(key))
                self.send_error(HTTPStatus.NOT_FOUND)
            except FileNotFoundError as error:
                self._send_error_json(error, HTTPStatus.NOT_FOUND)
            except Exception as error:
                self._send_error_json(error)

        def do_POST(self) -> None:
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
                if self.path == "/api/save":
                    key = str(payload["key"])
                    review = payload["review"]
                    return self._send_json(app.save_review_and_export(key, review))
                if self.path == "/api/auto-polygon":
                    return self._send_json(app.auto_polygon(payload))
                if self.path == "/api/tutorial/complete":
                    return self._send_json(app.mark_tutorial_completed(bool(payload.get("completed", True))))
                self.send_error(HTTPStatus.NOT_FOUND)
            except Exception as error:
                self._send_error_json(error)

        def _send_static(self, name: str) -> None:
            try:
                data = resources.files("rbccps_annotator.static").joinpath(name).read_bytes()
            except FileNotFoundError:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            mime = mimetypes.guess_type(name)[0] or "application/octet-stream"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_file(self, path: Path) -> None:
            data = path.read_bytes()
            mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}"
    print(f"RBCCPS annotator running at: {url}")
    print(f"Workspace: {app.workspace}")
    if open_browser:
        webbrowser.open(url)
    server.serve_forever()
