from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from rbccps_annotator.workspace import ensure_dir, load_manifest, load_review, write_json


def generate_confounder_augmentations(workspace: Path, output: Path, probability: float, variant: str, seed: int) -> Path:
    try:
        from PIL import Image, ImageDraw, ImageFilter
    except ImportError as error:
        raise RuntimeError("Pillow is required for augmentation generation. Install the project with the train optional dependencies.") from error

    rng = random.Random(seed)
    manifest = load_manifest(workspace)
    image_dir = ensure_dir(output / "images")
    metadata: list[dict[str, Any]] = []
    for item in manifest.get("items", []):
        if rng.random() > probability:
            continue
        review = load_review(workspace, item)
        polygons = [poly for poly in review.get("polygons", []) if poly.get("augmentation_allowed")]
        if not polygons:
            continue
        image_path = Path(item["image_path"])
        if not image_path.exists():
            continue
        image = Image.open(image_path).convert("RGB")
        boxes = [box.get("bbox_xyxy", []) for box in review.get("boxes", []) if box.get("bbox_xyxy")]
        applied: list[str] = []
        for polygon in polygons:
            points = [(float(x), float(y)) for x, y in polygon.get("points", [])]
            if len(points) < 3:
                continue
            margin = int(polygon.get("mask_exclusion_margin_px") or 12)
            if _polygon_intersects_any_box(points, boxes, margin):
                continue
            mask = Image.new("L", image.size, 0)
            draw = ImageDraw.Draw(mask)
            draw.polygon(points, fill=210)
            mask = mask.filter(ImageFilter.GaussianBlur(radius=3))
            image = _apply_variant(image, mask, variant)
            applied.append(str(polygon.get("polygon_id", "")))
        if not applied:
            continue
        output_name = f"{item['key']}_{variant}{image_path.suffix.lower()}"
        image.save(image_dir / output_name)
        metadata.append({"key": item["key"], "source_image": str(image_path), "augmented_image": str(image_dir / output_name), "variant": variant, "polygons": applied})
    write_json(output / "augmentation_manifest.json", {"workspace": str(workspace.resolve()), "probability": probability, "variant": variant, "items": metadata})
    return output


def _apply_variant(image: Any, mask: Any, variant: str) -> Any:
    from PIL import Image, ImageFilter

    if variant == "blur":
        return Image.composite(image.filter(ImageFilter.GaussianBlur(radius=8)), image, mask)
    if variant == "gray":
        gray = Image.new("RGB", image.size, (70, 70, 70))
        return Image.composite(gray, image, mask)
    if variant == "noise":
        noise = Image.effect_noise(image.size, 24).convert("RGB")
        return Image.composite(noise, image, mask)
    dim = image.point(lambda value: int(value * 0.35))
    return Image.composite(dim, image, mask)


def _polygon_intersects_any_box(points: list[tuple[float, float]], boxes: list[list[float]], margin: int) -> bool:
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
