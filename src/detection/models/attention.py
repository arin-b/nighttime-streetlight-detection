from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


FeatureMaps = torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...]


class GeometryAttentionBlock(nn.Module):
    """PDF-style geometry-aware attention using vertical and horizontal context."""

    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int = 7,
        neutral_bias: float = 4.0,
        config: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("Geometry attention channels must be positive.")
        if config is not None:
            kernel_size = int(config.get("kernel_size", kernel_size))
            neutral_bias = float(config.get("neutral_bias", neutral_bias))
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("Geometry attention kernel_size must be a positive odd integer.")

        padding = kernel_size // 2
        self.conv_v = nn.Conv2d(channels, channels, kernel_size=(kernel_size, 1), padding=(padding, 0))
        self.conv_h = nn.Conv2d(channels, channels, kernel_size=(1, kernel_size), padding=(0, padding))
        self.fuse = nn.Conv2d(2 * channels, channels, kernel_size=1)

        # Start close to an identity gate so adding the module does not shock pretrained YOLO weights.
        nn.init.zeros_(self.fuse.weight)
        nn.init.constant_(self.fuse.bias, neutral_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        vertical = self.conv_v(x)
        horizontal = self.conv_h(x)
        attention = torch.sigmoid(self.fuse(torch.cat([vertical, horizontal], dim=1)))
        return x * attention


class GeometryAttention(nn.Module):
    """Multi-scale geometry-aware attention for YOLO detect-head feature lists."""

    def __init__(
        self,
        channels: int | Sequence[int],
        *,
        kernel_size: int = 7,
        neutral_bias: float = 4.0,
        config: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        if config is not None:
            kernel_size = int(config.get("kernel_size", kernel_size))
            neutral_bias = float(config.get("neutral_bias", neutral_bias))
        self.channels = _normalise_channels(channels)
        self.blocks = nn.ModuleList(
            GeometryAttentionBlock(channel, kernel_size=kernel_size, neutral_bias=neutral_bias, config=config)
            for channel in self.channels
        )

    def forward(self, features: FeatureMaps) -> FeatureMaps:
        tensors, container = _feature_sequence(features)
        _validate_feature_count(tensors, self.blocks, "geometry attention")
        outputs = [block(feature) for block, feature in zip(self.blocks, tensors)]
        return _restore_feature_sequence(outputs, container)


class SuppressionBranch(nn.Module):
    """Predict a negative attention logit map for one YOLO feature scale."""

    def __init__(
        self,
        channels: int,
        *,
        reduction: int = 4,
        init_bias: float = -4.0,
        config: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("SuppressionBranch channels must be positive.")
        if config is not None:
            reduction = int(config.get("reduction", reduction))
            init_bias = float(config.get("init_bias", init_bias))
        hidden_channels = max(1, channels // max(1, reduction))
        self.conv1 = nn.Conv2d(channels, hidden_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        nn.init.constant_(self.conv2.bias, init_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.relu(self.conv1(x)))


class NegativeAttention(nn.Module):
    """Learned negative attention branch with supervised BCE mask loss."""

    def __init__(
        self,
        channels: int | Sequence[int],
        *,
        reduction: int = 4,
        init_bias: float = -4.0,
        config: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        if config is not None:
            reduction = int(config.get("reduction", reduction))
            init_bias = float(config.get("init_bias", init_bias))
        self.channels = _normalise_channels(channels)
        self.branches = nn.ModuleList(
            SuppressionBranch(channel, reduction=reduction, init_bias=init_bias, config=config)
            for channel in self.channels
        )
        self.last_logits: tuple[torch.Tensor, ...] = ()

    def forward(self, features: FeatureMaps, negative_mask: torch.Tensor | None = None) -> FeatureMaps:
        del negative_mask  # The target mask is used by mask_loss(); inference predicts its own mask.
        tensors, container = _feature_sequence(features)
        _validate_feature_count(tensors, self.branches, "negative attention")
        outputs: list[torch.Tensor] = []
        logits: list[torch.Tensor] = []
        for branch, feature in zip(self.branches, tensors):
            logit = branch(feature)
            logits.append(logit)
            outputs.append(feature * (1.0 - torch.sigmoid(logit)))
        self.last_logits = tuple(logits)
        return _restore_feature_sequence(outputs, container)

    def mask_loss(self, target_mask: torch.Tensor | None) -> torch.Tensor:
        """Return pixel BCE between predicted negative maps and annotated negative masks."""
        if not self.last_logits:
            return _zero_from_module(self)
        if target_mask is None:
            return _zero_from_tensor(self.last_logits[0])

        losses: list[torch.Tensor] = []
        for logits in self.last_logits:
            target = _as_batched_mask(target_mask, device=logits.device, dtype=logits.dtype)
            if target.shape[0] == 1 and logits.shape[0] != 1:
                target = target.expand(logits.shape[0], -1, -1, -1)
            if target.shape[0] != logits.shape[0]:
                raise ValueError(
                    f"Negative mask batch size {target.shape[0]} does not match feature batch size {logits.shape[0]}."
                )
            target = F.interpolate(target, size=logits.shape[-2:], mode="nearest").clamp(0.0, 1.0)
            losses.append(F.binary_cross_entropy_with_logits(logits, target))
        return torch.stack(losses).mean()


class NegativeMaskBlock(NegativeAttention):
    """Backward-compatible name for the learned negative attention branch."""


def _as_batched_mask(mask: torch.Tensor, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(1)
    elif mask.ndim != 4:
        raise ValueError("Negative masks must have shape HxW, BxHxW, or Bx1xHxW.")
    return mask.to(device=device, dtype=dtype)


def _normalise_channels(channels: int | Sequence[int]) -> tuple[int, ...]:
    if isinstance(channels, int):
        return (channels,)
    values = tuple(int(channel) for channel in channels)
    if not values:
        raise ValueError("At least one channel count is required.")
    if any(channel <= 0 for channel in values):
        raise ValueError("Channel counts must be positive.")
    return values


def _feature_sequence(features: FeatureMaps) -> tuple[list[torch.Tensor], type | None]:
    if isinstance(features, list):
        return features, list
    if isinstance(features, tuple):
        return list(features), tuple
    return [features], None


def _restore_feature_sequence(features: list[torch.Tensor], container: type | None) -> FeatureMaps:
    if container is list:
        return features
    if container is tuple:
        return tuple(features)
    return features[0]


def _validate_feature_count(features: list[torch.Tensor], modules: nn.ModuleList, name: str) -> None:
    if len(features) != len(modules):
        raise ValueError(f"Expected {len(modules)} {name} feature maps, got {len(features)}.")


def _zero_from_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.sum() * 0.0


def _zero_from_module(module: nn.Module) -> torch.Tensor:
    parameter = next(module.parameters(), None)
    if parameter is None:
        return torch.tensor(0.0)
    return parameter.sum() * 0.0


def candidate_negative_mask_paths(
    image_path: Path,
    mask_root: Path,
    *,
    suffixes: Iterable[str] = (".png", ".jpg", ".jpeg", ".npy", ".pt"),
) -> list[Path]:
    """Return likely mask paths for an image, including mirrored split layouts."""

    candidates: list[Path] = []
    for suffix in suffixes:
        candidates.append(mask_root / f"{image_path.stem}{suffix}")
        parts = image_path.parts
        if "images" in parts:
            image_index = parts.index("images")
            relative_parts = parts[image_index + 1 :]
            if relative_parts:
                candidates.append(mask_root / Path(*relative_parts).with_suffix(suffix))
    return list(dict.fromkeys(candidates))
