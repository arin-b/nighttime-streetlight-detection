from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision import models, transforms
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from ultralytics import YOLO

from rbccps_measurement.contracts.input_schema import ClipManifest
from rbccps_measurement.ingest.validation import validate_clip_manifest
from rbccps_measurement.models.registry import get_asset
from rbccps_measurement.pipeline import run_clip_to_directory


class ZeroDCE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.e_conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.e_conv2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.e_conv3 = nn.Conv2d(32, 32, 3, 1, 1)
        self.e_conv4 = nn.Conv2d(32, 32, 3, 1, 1)
        self.e_conv5 = nn.Conv2d(64, 32, 3, 1, 1)
        self.e_conv6 = nn.Conv2d(64, 32, 3, 1, 1)
        self.e_conv7 = nn.Conv2d(64, 24, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        curves = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        enhanced = x
        for curve in torch.split(curves, 3, dim=1):
            enhanced = enhanced + curve * (torch.pow(enhanced, 2) - enhanced)
        return torch.clamp(enhanced, 0, 1), curves


class RetinexDecom(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net1_conv0 = nn.Conv2d(4, 64, 9, 1, 4)
        layers: list[nn.Module] = []
        for _ in range(5):
            layers.extend([nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(inplace=True)])
        self.net1_convs = nn.Sequential(*layers)
        self.net1_recon = nn.Conv2d(64, 4, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        max_rgb, _ = torch.max(x, dim=1, keepdim=True)
        feat = F.relu(self.net1_conv0(torch.cat([x, max_rgb], 1)))
        feat = self.net1_convs(feat)
        out = torch.sigmoid(self.net1_recon(feat))
        return out[:, :3], out[:, 3:4]


def load_state(path: Path) -> dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    return obj


def image_tensor(path: Path, max_side: int = 384) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    arr = np.asarray(image).astype("float32") / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def save_rgb(tensor: torch.Tensor, path: Path) -> None:
    arr = tensor.detach().cpu().squeeze(0).clamp(0, 1).permute(1, 2, 0).numpy()
    Image.fromarray((arr * 255).astype("uint8")).save(path)


def save_gray(tensor: torch.Tensor, path: Path) -> None:
    arr = tensor.detach().cpu().squeeze().clamp(0, 1).numpy()
    Image.fromarray((arr * 255).astype("uint8")).save(path)


def luma_mean(tensor: torch.Tensor) -> float:
    return float((0.2126 * tensor[:, 0:1] + 0.7152 * tensor[:, 1:2] + 0.0722 * tensor[:, 2:3]).mean().item())


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> None:
    repo = repo_root()
    samples = json.loads((repo / "runs/measurement_pseudo_annotation_classes/selected_samples.json").read_text(encoding="utf-8-sig"))
    out_root = repo / "runs/measurement_all_slots_annotation_classes"
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "selected_samples.json").write_text(json.dumps(samples, indent=2), encoding="utf-8")

    paths = {name: repo / get_asset(name).local_path for name in [
        "streetlight_detector_v3",
        "segmentation_deeplabv3_mobilenet_v3",
        "feature_resnet18_imagenet",
        "lowlight_zero_dce_epoch99",
        "retinex_decom_9200",
    ]}

    yolo = YOLO(str(paths["streetlight_detector_v3"]))
    segmentation = deeplabv3_mobilenet_v3_large(weights=None, weights_backbone=None, num_classes=21, aux_loss=True).eval()
    segmentation.load_state_dict(load_state(paths["segmentation_deeplabv3_mobilenet_v3"]), strict=True)
    resnet = models.resnet18(weights=None).eval()
    resnet.load_state_dict(load_state(paths["feature_resnet18_imagenet"]), strict=True)
    feature_encoder = nn.Sequential(*list(resnet.children())[:-1]).eval()
    zero_dce = ZeroDCE().eval()
    zero_dce.load_state_dict(load_state(paths["lowlight_zero_dce_epoch99"]), strict=True)
    retinex = RetinexDecom().eval()
    retinex.load_state_dict(load_state(paths["retinex_decom_9200"]), strict=True)

    prep_resnet = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    prep_seg = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    palette = np.array([
        [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128],
        [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
        [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
        [0, 192, 0], [128, 192, 0], [0, 64, 128],
    ], dtype=np.uint8)

    try:
        font = ImageFont.truetype("arial.ttf", 20)
        small = ImageFont.truetype("arial.ttf", 15)
    except Exception:
        font = ImageFont.load_default()
        small = ImageFont.load_default()

    summary = []
    cards = []
    for sample in samples:
        sample_class = sample["sample_class"]
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in sample_class)
        sample_dir = out_root / safe
        frames_dir = sample_dir / "frames"
        slots_dir = sample_dir / "slot_outputs"
        measure_dir = sample_dir / "measurement"
        frames_dir.mkdir(parents=True, exist_ok=True)
        slots_dir.mkdir(parents=True, exist_ok=True)

        src = Path(sample["image_path"])
        frame_path = frames_dir / ("000001" + src.suffix.lower())
        shutil.copy2(src, frame_path)
        full = Image.open(frame_path).convert("RGB")
        width, height = full.size

        pred = yolo.predict(str(frame_path), conf=0.01, iou=0.55, max_det=8, verbose=False)[0]
        detections = []
        if pred.boxes is not None:
            for box in pred.boxes:
                detections.append((
                    [float(v) for v in box.xyxy[0].tolist()],
                    float(box.conf[0].item()),
                    int(box.cls[0].item()) if box.cls is not None else 0,
                ))
        detections.sort(key=lambda item: item[1], reverse=True)

        timestamp = 1_700_000_000_000_000_000
        tracks = []
        for index, (xyxy, score, class_id) in enumerate(detections, start=1):
            x1, y1, x2, y2 = xyxy
            x1 = max(0.0, min(width - 1.0, x1))
            y1 = max(0.0, min(height - 1.0, y1))
            x2 = max(x1 + 1.0, min(float(width), x2))
            y2 = max(y1 + 1.0, min(float(height), y2))
            tracks.append({
                "frame_id": 1,
                "timestamp_ns": timestamp,
                "track_id": f"yolo_lamp_{index}",
                "class_name": "streetlight_lamp_head",
                "bbox_xyxy": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                "bbox_format": "pixel_xyxy_original_frame",
                "detector_score": round(score, 4),
                "track_confidence": round(score, 4),
                "track_age": 1,
                "lost_count": 0,
                "source_model": "streetlight_detector_v3:hpc_pull",
                "optional_cue_scores": {"yolo_class_id": float(class_id), "detector_conf_threshold": 0.01},
            })

        with torch.no_grad():
            seg_map = segmentation(prep_seg(full).unsqueeze(0))["out"].argmax(1).squeeze(0).cpu().numpy().astype("uint8")
            uniq, counts = np.unique(seg_map, return_counts=True)
            seg_hist = {str(int(k)): round(float(v) / float(counts.sum()), 4) for k, v in zip(uniq, counts)}
            Image.fromarray(palette[seg_map % len(palette)]).save(slots_dir / "deeplabv3_segmentation.png")

            embedding = feature_encoder(prep_resnet(full).unsqueeze(0)).flatten(1).squeeze(0).cpu().numpy()
            low_tensor = image_tensor(frame_path)
            enhanced, curves = zero_dce(low_tensor)
            reflectance, illumination = retinex(low_tensor)
            save_rgb(enhanced, slots_dir / "zero_dce_enhanced.png")
            save_rgb(reflectance, slots_dir / "retinex_reflectance.png")
            save_gray(illumination, slots_dir / "retinex_illumination.png")

        slot_metrics = {
            "weights_used": {key: str(value) for key, value in paths.items()},
            "detector": {"implementation": "ultralytics.YOLO", "conf_threshold": 0.01, "tracks": len(tracks), "scores": [t["detector_score"] for t in tracks]},
            "segmentation_deeplabv3_mobilenet_v3": {"implementation": "torchvision.deeplabv3_mobilenet_v3_large", "segmentation_class_histogram": seg_hist, "artifact": "slot_outputs/deeplabv3_segmentation.png"},
            "feature_resnet18_imagenet": {"implementation": "torchvision.resnet18 trunk", "embedding_dim": int(embedding.shape[0]), "embedding_l2_norm": round(float(np.linalg.norm(embedding)), 4), "embedding_mean": round(float(embedding.mean()), 4), "embedding_std": round(float(embedding.std()), 4)},
            "lowlight_zero_dce_epoch99": {"implementation": "Zero-DCE enhance_net_nopool", "input_luma_mean": round(luma_mean(low_tensor), 4), "enhanced_luma_mean": round(luma_mean(enhanced), 4), "curve_mean": round(float(curves.mean().item()), 4), "artifact": "slot_outputs/zero_dce_enhanced.png"},
            "retinex_decom_9200": {"implementation": "RetinexNet DecomNet", "illumination_mean": round(float(illumination.mean().item()), 4), "illumination_std": round(float(illumination.std().item()), 4), "reflectance_mean": round(float(reflectance.mean().item()), 4), "artifacts": ["slot_outputs/retinex_reflectance.png", "slot_outputs/retinex_illumination.png"]},
        }
        (slots_dir / "slot_metrics.json").write_text(json.dumps(slot_metrics, indent=2), encoding="utf-8")

        manifest = {
            "clip_id": f"all_slots_{safe}",
            "device_id": "annotation_dataset_image_no_phone_metadata",
            "calibration_level": 1,
            "policy_id": "rbccps_measurement_policy_v1",
            "video_uri": None,
            "frames": [{
                "frame_id": 1,
                "timestamp_ns": timestamp,
                "image_uri": "frames/" + frame_path.name,
                "image_format": frame_path.suffix.lower().lstrip("."),
                "width": width,
                "height": height,
                "camera": {"exposure_time_s": None, "sensor_sensitivity_iso": None, "ae_mode": "auto", "hdr_mode": "unknown", "night_mode": True, "metadata_quality": "pseudo_from_static_dataset_image"},
                "pose": {"latitude": None, "longitude": None, "gps_accuracy_m": None, "heading_deg": None, "imu_quality": "missing"},
            }],
            "tracks": tracks,
            "optional_calibration": {
                "photometric": {"field_lux_calibration_id": None},
                "map_priors": {"learned_slot_metrics_uri": "slot_outputs/slot_metrics.json"},
            },
        }
        manifest_path = sample_dir / "clip_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        status = "no_detections"
        reports_count = 0
        first_score = None
        first_category = None
        error = None
        if tracks:
            try:
                validate_clip_manifest(ClipManifest.from_dict(manifest))
                reports = run_clip_to_directory(manifest_path, measure_dir, measurement_run_id=f"all_slots_run_{safe}")
                status = "measured"
                reports_count = len(reports)
                first = reports[0].to_dict()
                first_score = first["metrics"]["overall_useful_illumination_score"]
                first_category = first["metrics"]["overall_category"]
                (sample_dir / "all_slots_measurement_report.json").write_text(
                    json.dumps({"measurement_reports": [r.to_dict() for r in reports], "slot_metrics": slot_metrics}, indent=2),
                    encoding="utf-8",
                )
            except Exception as exc:
                status = "measurement_failed"
                error = repr(exc)

        summary.append({
            "sample_class": sample_class,
            "image_uid": sample["image_uid"],
            "annotation_count": int(sample["annotation_count"]),
            "detector_tracks": len(tracks),
            "max_detector_score": max([item["detector_score"] for item in tracks], default=None),
            "measurement_status": status,
            "reports": reports_count,
            "first_category": first_category,
            "first_score": first_score,
            "slot_metrics": str(slots_dir / "slot_metrics.json"),
            "manifest": str(manifest_path),
            "output": str(measure_dir) if tracks else None,
            "error": error,
        })

        thumb = full.copy()
        thumb.thumbnail((560, 315), Image.Resampling.LANCZOS)
        sx = thumb.width / width
        sy = thumb.height / height
        card = Image.new("RGB", (620, 430), (248, 248, 246))
        card.paste(thumb, (30, 56))
        draw = ImageDraw.Draw(card)
        draw.rectangle((0, 0, 619, 429), outline=(55, 55, 55), width=1)
        draw.text((24, 18), sample_class, fill=(15, 15, 15), font=font)
        for index, track in enumerate(tracks[:8]):
            x1, y1, x2, y2 = track["bbox_xyxy"]
            color = (255, 210, 0) if index == 0 else (0, 220, 255)
            rect = [30 + x1 * sx, 56 + y1 * sy, 30 + x2 * sx, 56 + y2 * sy]
            draw.rectangle(rect, outline=color, width=4)
            draw.text((rect[0] + 3, max(58, rect[1] - 18)), f"{track['track_id']} {track['detector_score']:.2f}", fill=color, font=small)
        y = 366
        for line in [
            f"image_uid: {sample['image_uid']}",
            f"GT boxes: {sample['annotation_count']} | YOLO tracks: {len(tracks)} | status: {status}",
            f"category: {first_category} | score: {first_score}",
            "slots: YOLO + DeepLab + ResNet + ZeroDCE + Retinex",
        ]:
            draw.text((30, y), line, fill=(30, 30, 30), font=small)
            y += 16
        card.save(sample_dir / "all_slots_measurement_visual.png")
        cards.append(card)

    (out_root / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    sheet = Image.new("RGB", (1240, 860), (235, 235, 232))
    for card, pos in zip(cards, [(0, 0), (620, 0), (0, 430), (620, 430)]):
        sheet.paste(card, pos)
    sheet.save(out_root / "all_slots_measurement_contact_sheet.png")
    print(json.dumps({"output_root": str(out_root), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
