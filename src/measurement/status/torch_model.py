from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from rbccps_measurement.contracts.module_io import LampCropSequence, LatentEmissionStateOutput
from rbccps_measurement.status.model import CAPTURE_CLASSES, EMISSION_CLASSES, FLICKER_CLASSES, OCCLUSION_CLASSES, STATUS_CLASSES, StatusModelConfig, _posterior


def torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def build_torch_status_model(config: StatusModelConfig | None = None):
    import torch
    import torch.nn as nn

    config = config or StatusModelConfig()

    class TinyStatusNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = config
            self.cnn = nn.Sequential(
                nn.Conv2d(3, 12, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(12, 24, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.token = nn.Sequential(nn.Linear(config.token_dim, 24), nn.ReLU())
            self.gru = nn.GRU(input_size=48, hidden_size=config.hidden_dim, batch_first=True)
            self.status = nn.Linear(config.hidden_dim, len(STATUS_CLASSES))
            self.emission = nn.Linear(config.hidden_dim, len(EMISSION_CLASSES))
            self.occlusion = nn.Linear(config.hidden_dim, len(OCCLUSION_CLASSES))
            self.capture = nn.Linear(config.hidden_dim, len(CAPTURE_CLASSES))
            self.flicker = nn.Linear(config.hidden_dim, len(FLICKER_CLASSES))
            self.confidence = nn.Linear(config.hidden_dim, 1)

        def forward(self, crops, tokens, mask):
            batch, time, height, width, channels = crops.shape
            x = crops.permute(0, 1, 4, 2, 3).reshape(batch * time, channels, height, width)
            crop_features = self.cnn(x).reshape(batch, time, 24)
            token_features = self.token(tokens)
            features = torch.cat([crop_features, token_features], dim=-1)
            output, _ = self.gru(features)
            lengths = mask.long().sum(dim=1).clamp(min=1)
            last_index = (lengths - 1).view(batch, 1, 1).expand(batch, 1, output.shape[-1])
            pooled = output.gather(1, last_index).squeeze(1)
            return {
                "status": self.status(pooled),
                "emission": self.emission(pooled),
                "occlusion": self.occlusion(pooled),
                "capture": self.capture(pooled),
                "flicker": self.flicker(pooled),
                "confidence": self.confidence(pooled).squeeze(-1),
            }

    return TinyStatusNet()


def load_torch_status_model(weights_path: str | Path, checkpoint: dict[str, Any]):
    import torch

    config_payload = checkpoint.get("model_config", {})
    config = StatusModelConfig(
        sequence_length=int(config_payload.get("sequence_length", 16)),
        crop_size=int(config_payload.get("crop_size", 64)),
        token_dim=int(config_payload.get("token_dim", 9)),
        hidden_dim=int(config_payload.get("hidden_dim", 64)),
    )
    model = build_torch_status_model(config)
    payload = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def _softmax_dict(values: np.ndarray, classes: tuple[str, ...]) -> dict[str, float]:
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    probs = exp / max(float(np.sum(exp)), 1e-9)
    return {label: round(float(value), 6) for label, value in zip(classes, probs)}


def predict_torch_status(model: Any, sequence: LampCropSequence) -> LatentEmissionStateOutput:
    import torch

    with torch.no_grad():
        crops = torch.from_numpy(sequence.crop_tensor[None].astype(np.float32))
        tokens = torch.from_numpy(sequence.metadata_tokens[None].astype(np.float32))
        mask = torch.from_numpy(sequence.valid_mask[None].astype(bool))
        outputs = model(crops, tokens, mask)
        status_logits = outputs["status"][0].detach().cpu().numpy()
        emission_logits = outputs["emission"][0].detach().cpu().numpy()
        occlusion_logits = outputs["occlusion"][0].detach().cpu().numpy()
        capture_logits = outputs["capture"][0].detach().cpu().numpy()
        flicker_logits = outputs["flicker"][0].detach().cpu().numpy()
        confidence = float(torch.sigmoid(outputs["confidence"][0]).detach().cpu())
    status = _softmax_dict(status_logits, STATUS_CLASSES)
    label = max(status, key=status.get)
    emission = _softmax_dict(emission_logits, EMISSION_CLASSES)
    occlusion = _softmax_dict(occlusion_logits, OCCLUSION_CLASSES)
    capture = _softmax_dict(capture_logits, CAPTURE_CLASSES)
    flicker = _softmax_dict(flicker_logits, FLICKER_CLASSES)
    return LatentEmissionStateOutput(
        track_id=sequence.track_id,
        status_label=label,
        status_posterior=status,
        emission_posterior=emission,
        occlusion_posterior=occlusion,
        capture_posterior=capture,
        flicker_posterior=flicker,
        dim_probability=status.get("dim", 0.0),
        occluded_probability=status.get("occluded", 0.0),
        flicker_index=max(status.get("flicker", 0.0), flicker.get("flicker", 0.0)),
        saturated_flag=status.get("saturated", 0.0) > 0.5 or capture.get("saturated", 0.0) > 0.5,
        confidence=confidence,
        quality_flags=tuple(sequence.quality_flags),
        metadata={"implementation": "torch_cnn_gru_status_v1"},
    )


def save_initialized_or_trained_model(path: str | Path, train_samples: list[tuple[LampCropSequence, str]], config: StatusModelConfig | None = None) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    config = config or StatusModelConfig()
    model = build_torch_status_model(config)
    label_to_id = {label: index for index, label in enumerate(STATUS_CLASSES)}
    trained_steps = 0
    if train_samples:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        for sequence, label in train_samples:
            if label not in label_to_id:
                continue
            crops = torch.from_numpy(sequence.crop_tensor[None].astype(np.float32))
            tokens = torch.from_numpy(sequence.metadata_tokens[None].astype(np.float32))
            mask = torch.from_numpy(sequence.valid_mask[None].astype(bool))
            target = torch.tensor([label_to_id[label]], dtype=torch.long)
            outputs = model(crops, tokens, mask)
            loss = F.cross_entropy(outputs["status"], target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            trained_steps += 1
        model.eval()
    torch.save({"state_dict": model.state_dict(), "status_classes": STATUS_CLASSES, "trained_steps": trained_steps}, path)
    return {"trained_steps": trained_steps, "status_classes": list(STATUS_CLASSES)}
