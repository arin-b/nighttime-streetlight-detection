from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageFilter

from rbccps_annotator.segmenter import propose_mask_polygon, status as segmenter_status

PROMPT_SEGMENTER_ROOT = Path("models") / "annotator" / "prompt_segmenter"


@dataclass(frozen=True)
class AutoPolygonResult:
    points: list[list[float]]
    engine: str
    confidence: float
    warnings: list[str]
    model_status: str
    bbox_xyxy: list[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "points": self.points,
            "engine": self.engine,
            "confidence": self.confidence,
            "warnings": self.warnings,
            "model_status": self.model_status,
            "bbox_xyxy": self.bbox_xyxy,
        }


def prompt_segmenter_status(repo_root: Path | None = None) -> dict[str, Any]:
    return segmenter_status(repo_root).to_dict()


def propose_auto_polygon(
    image_path: Path,
    bbox_xyxy: list[float],
    protected_boxes: list[list[float]] | None = None,
    margin_px: int = 12,
    repo_root: Path | None = None,
) -> AutoPolygonResult:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    bbox = _clamp_box(bbox_xyxy, width, height)
    warnings: list[str] = []

    if _box_too_small(bbox):
        raise ValueError("Smart Surface drag area is too small. Drag a larger rectangle around the surface.")

    status = prompt_segmenter_status(repo_root)
    if not status["ready"]:
        warnings.append("AI shape tool is not installed yet. Using the rough shape helper.")

    points, engine, confidence = propose_mask_polygon(image_path, bbox, repo_root)
    if not points:
        points, engine, confidence = _opencv_grabcut_polygon(image_path, bbox)
    if not points:
        points = _edge_guided_polygon(image, bbox)
        engine = "pil_edge_fallback"
        confidence = 0.35
    if not points:
        points = _box_polygon(bbox)
        engine = "rectangle_fallback"
        confidence = 0.2

    if _polygon_intersects_any_box(points, protected_boxes or [], margin_px):
        warnings.append(f"This shape touches a lamp box plus {margin_px}px safety margin. Check before keeping.")

    return AutoPolygonResult(
        points=[[round(float(x), 2), round(float(y), 2)] for x, y in points],
        engine=engine,
        confidence=confidence,
        warnings=warnings,
        model_status="ready" if status["ready"] else "missing",
        bbox_xyxy=bbox,
    )


def _clamp_box(box: list[float], width: int, height: int) -> list[float]:
    if len(box) != 4:
        raise ValueError("bbox_xyxy must contain four values.")
    x1, y1, x2, y2 = [float(value) for value in box]
    left = max(0.0, min(x1, x2))
    top = max(0.0, min(y1, y2))
    right = min(float(width - 1), max(x1, x2))
    bottom = min(float(height - 1), max(y1, y2))
    return [left, top, right, bottom]


def _box_too_small(box: list[float]) -> bool:
    return (box[2] - box[0]) < 8 or (box[3] - box[1]) < 8


def _box_polygon(box: list[float]) -> list[list[float]]:
    x1, y1, x2, y2 = box
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def _opencv_grabcut_polygon(image_path: Path, box: list[float]) -> tuple[list[list[float]], str, float]:
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except Exception:
        return [], "", 0.0

    image = cv2.imread(str(image_path))
    if image is None:
        return [], "", 0.0
    x1, y1, x2, y2 = [int(round(value)) for value in box]
    rect = (x1, y1, max(1, x2 - x1), max(1, y2 - y1))
    mask = np.zeros(image.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_RECT)
    except Exception:
        return [], "", 0.0
    binary = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype("uint8")
    contours, _hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [], "", 0.0
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area < 20:
        return [], "", 0.0
    epsilon = max(2.0, 0.018 * cv2.arcLength(contour, True))
    approx = cv2.approxPolyDP(contour, epsilon, True)
    points = [[float(point[0][0]), float(point[0][1])] for point in approx]
    if len(points) < 3:
        return [], "", 0.0
    box_area = max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
    confidence = max(0.25, min(0.72, area / box_area))
    return points[:80], "opencv_grabcut_fallback", round(confidence, 3)


def _edge_guided_polygon(image: Image.Image, box: list[float]) -> list[list[float]]:
    x1, y1, x2, y2 = [int(round(value)) for value in box]
    crop = image.crop((x1, y1, x2, y2)).convert("L").filter(ImageFilter.FIND_EDGES)
    if crop.width < 2 or crop.height < 2:
        return []
    pixels = crop.load()
    threshold = max(12, int(_mean_luma(crop) * 1.2))
    xs: list[int] = []
    ys: list[int] = []
    for y in range(crop.height):
        for x in range(crop.width):
            if pixels[x, y] >= threshold:
                xs.append(x)
                ys.append(y)
    if not xs or not ys:
        return []
    pad = 3
    left = max(0, min(xs) - pad) + x1
    right = min(crop.width - 1, max(xs) + pad) + x1
    top = max(0, min(ys) - pad) + y1
    bottom = min(crop.height - 1, max(ys) + pad) + y1
    if right - left < 8 or bottom - top < 8:
        return []
    return [[left, top], [right, top], [right, bottom], [left, bottom]]


def _mean_luma(image: Image.Image) -> float:
    hist = image.histogram()
    total = sum(hist) or 1
    return sum(index * count for index, count in enumerate(hist)) / total


def _polygon_intersects_any_box(points: list[list[float]], boxes: list[list[float]], margin: int) -> bool:
    if not points:
        return False
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    px1, py1, px2, py2 = min(xs), min(ys), max(xs), max(ys)
    for box in boxes:
        if len(box) != 4:
            continue
        x1, y1, x2, y2 = [float(value) for value in box]
        x1 -= margin
        y1 -= margin
        x2 += margin
        y2 += margin
        if px1 <= x2 and px2 >= x1 and py1 <= y2 and py2 >= y1:
            return True
    return False
