from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from rbccps_od.models.checkpoint_resolver import resolve_checkpoint


class RetinexDecompositionModel:
    def __init__(
        self,
        checkpoint: str | Path | None = None,
        enabled: bool = False,
        asset_name: str = "retinex_decom_9200",
    ) -> None:
        self.checkpoint = Path(checkpoint) if checkpoint else None
        self.enabled = enabled
        self.asset_name = asset_name
        self._model = None
        self._device = None

    def _import_torch(self):
        import torch
        import torch.nn as nn

        class DecomNet(nn.Module):
            def __init__(self, channel: int = 64, kernel_size: int = 3):
                super().__init__()
                self.net1_conv0 = nn.Conv2d(4, channel, kernel_size * 3, padding=4, padding_mode="replicate")
                self.net1_convs = nn.Sequential(
                    nn.Conv2d(channel, channel, kernel_size, padding=1, padding_mode="replicate"),
                    nn.ReLU(),
                    nn.Conv2d(channel, channel, kernel_size, padding=1, padding_mode="replicate"),
                    nn.ReLU(),
                    nn.Conv2d(channel, channel, kernel_size, padding=1, padding_mode="replicate"),
                    nn.ReLU(),
                    nn.Conv2d(channel, channel, kernel_size, padding=1, padding_mode="replicate"),
                    nn.ReLU(),
                    nn.Conv2d(channel, channel, kernel_size, padding=1, padding_mode="replicate"),
                    nn.ReLU(),
                )
                self.net1_recon = nn.Conv2d(channel, 4, kernel_size, padding=1, padding_mode="replicate")

            def forward(self, input_im):
                input_max = torch.max(input_im, dim=1, keepdim=True)[0]
                input_img = torch.cat((input_max, input_im), dim=1)
                feats0 = self.net1_conv0(input_img)
                feats = self.net1_convs(feats0)
                outs = self.net1_recon(feats)
                reflectance = torch.sigmoid(outs[:, 0:3, :, :])
                illumination = torch.sigmoid(outs[:, 3:4, :, :])
                return reflectance, illumination

        return torch, DecomNet

    def _resolved_checkpoint(self) -> Path:
        if self.checkpoint is not None:
            return self.checkpoint.resolve()
        return resolve_checkpoint(self.asset_name, allow_missing=False)  # type: ignore[return-value]

    def _load_model(self, device: str | None = None):
        torch, DecomNet = self._import_torch()
        target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self._model is not None and self._device == str(target_device):
            return torch, self._model, target_device
        model = DecomNet().to(target_device)
        state = torch.load(self._resolved_checkpoint(), map_location=target_device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)
        model.eval()
        self._model = model
        self._device = str(target_device)
        return torch, model, target_device

    def decompose(self, frame_path: str | Path, output_dir: str | Path | None = None, device: str | None = None) -> dict[str, str]:
        if not self.enabled:
            return {"reflectance": str(frame_path), "illumination": str(frame_path)}
        torch, model, target_device = self._load_model(device=device)
        image = Image.open(frame_path).convert("RGB")
        data = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(data).permute(2, 0, 1).unsqueeze(0).to(target_device)
        with torch.no_grad():
            reflectance, illumination = model(tensor)
        reflectance_np = reflectance.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
        illum_np = illumination.squeeze(0).detach().cpu().squeeze(0).numpy()
        reflectance_np = np.clip(reflectance_np * 255.0, 0, 255).astype("uint8")
        illum_rgb = np.clip(np.stack([illum_np, illum_np, illum_np], axis=-1) * 255.0, 0, 255).astype("uint8")
        stem = Path(frame_path).stem
        out_root = Path(output_dir) if output_dir else Path(frame_path).parent
        out_root.mkdir(parents=True, exist_ok=True)
        reflectance_path = out_root / f"{stem}_reflectance.png"
        illumination_path = out_root / f"{stem}_illumination.png"
        Image.fromarray(reflectance_np).save(reflectance_path)
        Image.fromarray(illum_rgb).save(illumination_path)
        return {"reflectance": str(reflectance_path), "illumination": str(illumination_path)}
