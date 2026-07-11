from __future__ import annotations

import os
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROMPT_SEGMENTER_ROOT = Path("models") / "annotator" / "prompt_segmenter"

SAM2_ENGINE = "SAM2.1-Hiera-Tiny"
SAM2_FILENAME = "sam2.1_hiera_tiny.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
SAM2_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"

FASTSAM_ENGINE = "FastSAM-s"
FASTSAM_FILENAME = "FastSAM-s.pt"

_SAM2_PREDICTOR: Any | None = None
_SAM2_PATH: Path | None = None
_SAM2_DEVICE: str | None = None
_FASTSAM_MODEL: Any | None = None
_FASTSAM_PATH: Path | None = None


@dataclass(frozen=True)
class SegmenterStatus:
    root: Path
    weights_path: Path
    weights_present: bool
    package_available: bool
    ready: bool
    engine: str
    active_engine: str
    sam2_weights_path: Path
    sam2_weights_present: bool
    sam2_package_available: bool
    fastsam_weights_path: Path
    fastsam_weights_present: bool
    fastsam_package_available: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "weights_path": str(self.weights_path),
            "weights_present": self.weights_present,
            "package_available": self.package_available,
            "ready": self.ready,
            "engine": self.engine,
            "active_engine": self.active_engine,
            "sam2_weights_path": str(self.sam2_weights_path),
            "sam2_weights_present": self.sam2_weights_present,
            "sam2_package_available": self.sam2_package_available,
            "fastsam_weights_path": str(self.fastsam_weights_path),
            "fastsam_weights_present": self.fastsam_weights_present,
            "fastsam_package_available": self.fastsam_package_available,
        }


def segmenter_root(repo_root: Path | None = None) -> Path:
    return (repo_root or Path.cwd()) / PROMPT_SEGMENTER_ROOT


def sam2_weights_path(repo_root: Path | None = None) -> Path:
    return segmenter_root(repo_root) / SAM2_FILENAME


def fastsam_weights_path(repo_root: Path | None = None) -> Path:
    return segmenter_root(repo_root) / FASTSAM_FILENAME


def status(repo_root: Path | None = None) -> SegmenterStatus:
    _prepare_ultralytics_config(repo_root)
    sam2_weights = sam2_weights_path(repo_root)
    fastsam_weights = fastsam_weights_path(repo_root)
    sam2_available = _module_available("sam2")
    fastsam_available = _module_available("ultralytics")
    sam2_ready = sam2_weights.exists() and sam2_available
    fastsam_ready = fastsam_weights.exists() and fastsam_available
    active_engine = SAM2_ENGINE if sam2_ready else FASTSAM_ENGINE if fastsam_ready else ""
    return SegmenterStatus(
        root=segmenter_root(repo_root),
        weights_path=sam2_weights,
        weights_present=sam2_weights.exists(),
        package_available=sam2_available,
        ready=sam2_ready or fastsam_ready,
        engine=SAM2_ENGINE,
        active_engine=active_engine,
        sam2_weights_path=sam2_weights,
        sam2_weights_present=sam2_weights.exists(),
        sam2_package_available=sam2_available,
        fastsam_weights_path=fastsam_weights,
        fastsam_weights_present=fastsam_weights.exists(),
        fastsam_package_available=fastsam_available,
    )


def download_sam2(repo_root: Path | None = None, force: bool = False) -> Path:
    target = sam2_weights_path(repo_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not force:
        return target
    with urllib.request.urlopen(SAM2_URL) as response:
        target.write_bytes(response.read())
    if not target.exists() or target.stat().st_size < 1_000_000:
        raise RuntimeError(f"SAM2 download failed or produced a tiny file: {target}")
    return target


def download_fastsam(repo_root: Path | None = None, force: bool = False) -> Path:
    target = fastsam_weights_path(repo_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not force:
        return target

    _prepare_ultralytics_config(repo_root)
    try:
        from ultralytics import FastSAM  # type: ignore
    except Exception as error:
        raise RuntimeError("Ultralytics is required to download FastSAM-s. Use the project YOLO venv.") from error

    previous_cwd = Path.cwd()
    try:
        os.chdir(target.parent)
        FastSAM(FASTSAM_FILENAME)
    finally:
        os.chdir(previous_cwd)

    downloaded = target.parent / FASTSAM_FILENAME
    if downloaded.exists():
        return downloaded

    candidates = sorted(target.parent.rglob(FASTSAM_FILENAME))
    if candidates:
        shutil.copy2(candidates[0], target)
        return target
    raise RuntimeError(f"FastSAM download completed but {target} was not found.")


def download_segmenter(repo_root: Path | None = None, force: bool = False, engine: str = "sam2") -> list[Path]:
    selected = engine.lower()
    paths: list[Path] = []
    if selected in {"sam2", "both"}:
        paths.append(download_sam2(repo_root, force))
    if selected in {"fastsam", "both"}:
        paths.append(download_fastsam(repo_root, force))
    if not paths:
        raise ValueError("engine must be one of: sam2, fastsam, both")
    return paths


def propose_mask_polygon(image_path: Path, bbox_xyxy: list[float], repo_root: Path | None = None) -> tuple[list[list[float]], str, float]:
    segmenter_status = status(repo_root)
    if not segmenter_status.ready:
        return [], "", 0.0

    if segmenter_status.sam2_weights_present and segmenter_status.sam2_package_available:
        try:
            polygon, confidence = _predict_sam2_polygon(image_path, bbox_xyxy, segmenter_status.sam2_weights_path)
            if len(polygon) >= 3:
                return polygon, "sam2_1_hiera_tiny", confidence
        except Exception:
            pass

    if segmenter_status.fastsam_weights_present and segmenter_status.fastsam_package_available:
        try:
            polygon, confidence = _predict_fastsam_polygon(image_path, bbox_xyxy, segmenter_status.fastsam_weights_path)
            if len(polygon) >= 3:
                return polygon, "fastsam_s", confidence
        except Exception:
            pass

    return [], "", 0.0


def _predict_sam2_polygon(image_path: Path, bbox_xyxy: list[float], weights_path: Path) -> tuple[list[list[float]], float]:
    import numpy as np  # type: ignore
    import torch  # type: ignore
    from PIL import Image

    predictor = _load_sam2(weights_path)
    image_np = np.array(Image.open(image_path).convert("RGB"))
    box_np = np.array(bbox_xyxy, dtype=np.float32)
    with torch.inference_mode():
        predictor.set_image(image_np)
        masks, scores, _logits = predictor.predict(box=box_np, multimask_output=True)
    if masks is None or len(masks) == 0:
        return [], 0.0
    best_index = int(np.argmax(scores)) if scores is not None and len(scores) else 0
    polygon = _mask_to_polygon(masks[best_index])
    confidence = float(scores[best_index]) if scores is not None and len(scores) else 0.75
    return polygon, round(max(0.0, min(0.99, confidence)), 3)


def _load_sam2(weights_path: Path) -> Any:
    global _SAM2_PREDICTOR, _SAM2_PATH, _SAM2_DEVICE
    import torch  # type: ignore
    from sam2.build_sam import build_sam2  # type: ignore
    from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if _SAM2_PREDICTOR is not None and _SAM2_PATH == weights_path and _SAM2_DEVICE == device:
        return _SAM2_PREDICTOR
    model = build_sam2(SAM2_CONFIG, str(weights_path), device=device)
    _SAM2_PREDICTOR = SAM2ImagePredictor(model)
    _SAM2_PATH = weights_path
    _SAM2_DEVICE = device
    return _SAM2_PREDICTOR


def _predict_fastsam_polygon(image_path: Path, bbox_xyxy: list[float], weights_path: Path) -> tuple[list[list[float]], float]:
    _prepare_ultralytics_config()
    model = _load_fastsam(weights_path)
    try:
        result = model(str(image_path), bboxes=[bbox_xyxy], verbose=False, retina_masks=True)[0]
    except TypeError:
        result = model(str(image_path), bboxes=[bbox_xyxy], verbose=False)[0]

    masks = getattr(result, "masks", None)
    if masks is None:
        return [], 0.0

    polygon = _polygon_from_ultralytics_masks(masks)
    if len(polygon) < 3:
        return [], 0.0
    return polygon, _mask_confidence(masks, bbox_xyxy)


def _load_fastsam(weights_path: Path) -> Any:
    global _FASTSAM_MODEL, _FASTSAM_PATH
    if _FASTSAM_MODEL is not None and _FASTSAM_PATH == weights_path:
        return _FASTSAM_MODEL
    from ultralytics import FastSAM  # type: ignore

    _FASTSAM_MODEL = FastSAM(str(weights_path))
    _FASTSAM_PATH = weights_path
    return _FASTSAM_MODEL


def _prepare_ultralytics_config(repo_root: Path | None = None) -> None:
    config_dir = (repo_root or Path.cwd()) / ".cache" / "ultralytics"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(config_dir)


def _polygon_from_ultralytics_masks(masks: Any) -> list[list[float]]:
    xy = getattr(masks, "xy", None)
    if xy:
        longest = max(xy, key=lambda item: len(item))
        points = [[float(point[0]), float(point[1])] for point in longest]
        return _simplify_points(points, max_points=36)
    data = getattr(masks, "data", None)
    if data is None or len(data) == 0:
        return []
    mask = data[0]
    try:
        mask_np = mask.detach().cpu().numpy()
    except Exception:
        mask_np = mask
    return _mask_to_polygon(mask_np)


def _mask_to_polygon(mask_np: Any) -> list[list[float]]:
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except Exception:
        return []
    binary = (mask_np > 0.5).astype("uint8") * 255
    contours, _hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    if float(cv2.contourArea(contour)) < 20:
        return []
    epsilon = max(3.0, 0.026 * cv2.arcLength(contour, True))
    approx = cv2.approxPolyDP(contour, epsilon, True)
    return [[float(point[0][0]), float(point[0][1])] for point in approx][:36]


def _simplify_points(points: list[list[float]], max_points: int) -> list[list[float]]:
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore

        contour = np.array(points, dtype="float32").reshape((-1, 1, 2))
        epsilon = max(3.0, 0.024 * cv2.arcLength(contour, True))
        approx = cv2.approxPolyDP(contour, epsilon, True)
        simplified = [[float(point[0][0]), float(point[0][1])] for point in approx]
        if len(simplified) >= 3:
            points = simplified
    except Exception:
        pass
    if len(points) <= max_points:
        return points
    step = max(1, len(points) // max_points)
    return points[::step][:max_points]


def _mask_confidence(masks: Any, bbox_xyxy: list[float]) -> float:
    data = getattr(masks, "data", None)
    if data is None or len(data) == 0:
        return 0.78
    try:
        mask_np = data[0].detach().cpu().numpy()
        area = float((mask_np > 0.5).sum())
    except Exception:
        return 0.78
    x1, y1, x2, y2 = [float(value) for value in bbox_xyxy]
    box_area = max(1.0, (x2 - x1) * (y2 - y1))
    return round(max(0.35, min(0.94, area / box_area)), 3)


def _module_available(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False
