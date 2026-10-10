from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from rbccps_od.models.checkpoint_resolver import resolve_checkpoint


class LowLightEnhancer:
    def __init__(
        self,
        checkpoint: str | Path | None = None,
        enabled: bool = False,
        asset_name: str = "zero_dce_epoch99",
    ) -> None:
        self.checkpoint = Path(checkpoint) if checkpoint else None
        self.enabled = enabled
        self.asset_name = asset_name
        self._model = None
        self._device = None

    def _import_torch(self):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        class EnhanceNetNoPool(nn.Module):
            def __init__(self):
                super().__init__()
                self.relu = nn.ReLU(inplace=True)
                number_f = 32
                self.e_conv1 = nn.Conv2d(3, number_f, 3, 1, 1, bias=True)
                self.e_conv2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
                self.e_conv3 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
                self.e_conv4 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
                self.e_conv5 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
                self.e_conv6 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
                self.e_conv7 = nn.Conv2d(number_f * 2, 24, 3, 1, 1, bias=True)

            def forward(self, x):
                x1 = self.relu(self.e_conv1(x))
                x2 = self.relu(self.e_conv2(x1))
                x3 = self.relu(self.e_conv3(x2))
                x4 = self.relu(self.e_conv4(x3))
                x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
                x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
                x_r = F.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
                r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_r, 3, dim=1)
                x = x + r1 * (torch.pow(x, 2) - x)
                x = x + r2 * (torch.pow(x, 2) - x)
                x = x + r3 * (torch.pow(x, 2) - x)
                enhance_image_1 = x + r4 * (torch.pow(x, 2) - x)
                x = enhance_image_1 + r5 * (torch.pow(enhance_image_1, 2) - enhance_image_1)
                x = x + r6 * (torch.pow(x, 2) - x)
                x = x + r7 * (torch.pow(x, 2) - x)
                enhance_image = x + r8 * (torch.pow(x, 2) - x)
                return enhance_image_1, enhance_image, torch.cat([r1, r2, r3, r4, r5, r6, r7, r8], 1)

        return torch, EnhanceNetNoPool

    def _resolved_checkpoint(self) -> Path:
        if self.checkpoint is not None:
            return self.checkpoint.resolve()
        return resolve_checkpoint(self.asset_name, allow_missing=False)  # type: ignore[return-value]

    def _load_model(self, device: str | None = None):
        torch, EnhanceNetNoPool = self._import_torch()
        target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self._model is not None and self._device == str(target_device):
            return torch, self._model, target_device
        model = EnhanceNetNoPool().to(target_device)
        state = torch.load(self._resolved_checkpoint(), map_location=target_device)
        model.load_state_dict(state)
        model.eval()
        self._model = model
        self._device = str(target_device)
        return torch, model, target_device

    def enhance(self, frame_path: str | Path, output_path: str | Path | None = None, device: str | None = None) -> str:
        if not self.enabled:
            return str(frame_path)
        torch, model, target_device = self._load_model(device=device)
        image = Image.open(frame_path).convert("RGB")
        data = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(data).permute(2, 0, 1).unsqueeze(0).to(target_device)
        with torch.no_grad():
            _, enhanced, _ = model(tensor)
        enhanced_np = enhanced.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
        enhanced_np = np.clip(enhanced_np * 255.0, 0, 255).astype("uint8")
        target = Path(output_path) if output_path else Path(frame_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(enhanced_np).save(target)
        return str(target)
