from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any
import weakref

import torch

try:
    from ultralytics.models.yolo.detect import DetectionTrainer as _DetectionTrainer
except ImportError:
    try:
        from ultralytics.models.yolo.detect.train import DetectionTrainer as _DetectionTrainer
    except ImportError:
        _DetectionTrainer = object


@dataclass(frozen=True)
class AppliedAblationModules:
    cse_replacements: int = 0
    attention_heads: int = 0
    negative_mask_hooks: int = 0


class CSEOutputHook:
    def __call__(self, module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
        if hasattr(output, "ndim") and output.ndim == 4:
            return module.rbccps_cse(output)
        return output


class DetectAttentionInputHook:
    def __call__(self, module: Any, args: tuple[Any, ...]) -> tuple[Any, ...] | None:
        if not args:
            return None
        features = args[0]
        if module._rbccps_use_negative_attention and hasattr(module, "rbccps_negative_attention"):
            features = module.rbccps_negative_attention(features)
        if module._rbccps_use_geometry_attention and hasattr(module, "rbccps_geometry_attention"):
            features = module.rbccps_geometry_attention(features)
        return (features, *args[1:])


class NegativeMaskRootPreHook:
    def __init__(self, detect_heads: list[Any]) -> None:
        self.detect_heads = detect_heads

    def __call__(self, _module: Any, args: tuple[Any, ...]) -> None:
        payload = args[0] if args else None
        negative_mask = _extract_negative_mask(payload)
        for head in self.detect_heads:
            head._rbccps_negative_mask = negative_mask


class NegativeMaskRootPostHook:
    def __init__(self, detect_heads: list[Any]) -> None:
        self.detect_heads = detect_heads

    def __call__(self, _module: Any, _args: tuple[Any, ...], _output: Any) -> None:
        for head in self.detect_heads:
            head._rbccps_negative_mask = None


class NegativeAttentionLossWrapper:
    """Add supervised negative-attention BCE to the native YOLO detection loss."""

    def __init__(self, base_criterion: Any, model: Any, loss_weight: float) -> None:
        self.base_criterion = base_criterion
        self._model_ref = weakref.ref(model)
        self.loss_weight = float(loss_weight)

    def __call__(self, preds: Any, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        loss, loss_items = self.base_criterion(preds, batch)
        model = self._model_ref() if self._model_ref is not None else None
        mask_loss = collect_negative_attention_loss(model, batch.get("negative_mask")) if model is not None else loss.sum() * 0.0
        weighted_mask_loss = mask_loss * self.loss_weight
        batch_size = int(batch["img"].shape[0]) if isinstance(batch, dict) and "img" in batch else 1
        mask_component = (weighted_mask_loss * batch_size).reshape(1)
        loss = torch.cat([loss.reshape(-1), mask_component])
        loss_items = torch.cat([loss_items.reshape(-1), weighted_mask_loss.detach().reshape(1)])
        return loss, loss_items

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_model_ref"] = None
        return state

    def update(self) -> None:
        update = getattr(self.base_criterion, "update", None)
        if update is not None:
            update()


def replace_c2f_with_cse(module: Any) -> int:
    """Add CSE gates to C2f outputs without discarding pretrained C2f weights."""

    return _attach_cse_to_modules(module, {"C2f"})


def apply_ablation_modules(
    pytorch_model: Any,
    *,
    use_geometry_attention: bool = False,
    use_cse: bool = False,
    use_negative_attention: bool = False,
    negative_mask_loss_weight: float = 1.0,
    attention_config: dict[str, Any] | None = None,
) -> AppliedAblationModules:
    cse_replacements = 0
    if use_cse:
        cse_replacements = replace_c2f_with_cse(pytorch_model)
        cse_replacements += attach_cse_to_csp_modules(pytorch_model)
    attention_heads = 0
    detect_heads = _detect_heads(pytorch_model)
    if use_geometry_attention or use_negative_attention:
        for head in detect_heads:
            _patch_detect_head(
                head,
                use_geometry_attention=use_geometry_attention,
                use_negative_attention=use_negative_attention,
                attention_config=attention_config,
            )
        attention_heads = len(detect_heads)

    negative_mask_hooks = 0
    if use_negative_attention and detect_heads:
        pytorch_model._rbccps_negative_mask_loss_weight = float(negative_mask_loss_weight)
        negative_mask_hooks = len(detect_heads)

    if use_cse and not (
        cse_replacements or any(getattr(module, "_rbccps_cse_attached", False) for module in pytorch_model.modules())
    ):
        raise RuntimeError("CSE ablation requested, but no compatible CSP/C2f modules were found.")
    if (use_geometry_attention or use_negative_attention) and not attention_heads:
        raise RuntimeError("Attention ablation requested, but no detection head was found.")

    return AppliedAblationModules(
        cse_replacements=cse_replacements,
        attention_heads=attention_heads,
        negative_mask_hooks=negative_mask_hooks,
    )


def assert_ablation_modules(
    pytorch_model: Any,
    *,
    use_geometry_attention: bool,
    use_cse: bool,
    use_negative_attention: bool,
) -> None:
    """Fail if a loaded checkpoint does not contain the exact trained ablation."""

    cse_modules = [module for module in pytorch_model.modules() if getattr(module, "_rbccps_cse_attached", False)]
    if bool(cse_modules) != use_cse:
        raise RuntimeError(f"CSE checkpoint mismatch: expected={use_cse}, found={bool(cse_modules)}")
    if any(not any(isinstance(hook, CSEOutputHook) for hook in module._forward_hooks.values()) for module in cse_modules):
        raise RuntimeError("CSE checkpoint has a CSE block without its forward hook.")

    heads = _detect_heads(pytorch_model)
    for label, expected, module_name, flag_name in (
        ("geometry", use_geometry_attention, "rbccps_geometry_attention", "_rbccps_use_geometry_attention"),
        ("negative", use_negative_attention, "rbccps_negative_attention", "_rbccps_use_negative_attention"),
    ):
        found = bool(heads) and all(
            hasattr(head, module_name) and getattr(head, flag_name, False) and _has_attention_input_hook(head)
            for head in heads
        )
        if found != expected:
            raise RuntimeError(f"{label} attention checkpoint mismatch: expected={expected}, found={found}")


def attach_cse_to_csp_modules(module: Any) -> int:
    """Attach CSE blocks to YOLO26 CSP-style modules that are not C2f."""

    return _attach_cse_to_modules(module, {"C3k2", "C3k"})


def _attach_cse_to_modules(module: Any, module_names: set[str]) -> int:
    from rbccps_od.domain.cse import CSEBlock

    attached = 0
    for child in module.modules():
        if child.__class__.__name__ not in module_names:
            continue
        if getattr(child, "_rbccps_cse_attached", False):
            continue
        channels = _module_output_channels(child)
        if channels is None:
            continue
        child.rbccps_cse = CSEBlock(channels)
        child.register_forward_hook(CSEOutputHook())
        child._rbccps_cse_attached = True
        attached += 1
    return attached


def build_yolo26_ablation_model(
    weights_path: str | Path,
    *,
    use_geometry_attention: bool = False,
    use_cse: bool = False,
    use_negative_attention: bool = False,
    negative_mask_loss_weight: float = 1.0,
    attention_config: dict[str, Any] | None = None,
) -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("ultralytics is not installed. Install the training extras before running ablations.") from exc

    yolo = YOLO(str(weights_path))
    summary = apply_ablation_modules(
        yolo.model,
        use_geometry_attention=use_geometry_attention,
        use_cse=use_cse,
        use_negative_attention=use_negative_attention,
        negative_mask_loss_weight=negative_mask_loss_weight,
        attention_config=attention_config,
    )
    yolo.rbccps_ablation_modules = summary
    return yolo


def negative_mask_trainer(mask_root: str | Path | None, *, loss_weight: float = 1.0):
    """Return an importable trainer with external negative-mask supervision."""

    return ablation_trainer(use_negative_attention=True, mask_root=mask_root, loss_weight=loss_weight)


class RBCCPSAblationTrainer(_DetectionTrainer):
    """Construct the actual trainer model with its ablation modules installed."""

    def __init__(
        self,
        *args: Any,
        use_geometry_attention: bool = False,
        use_cse: bool = False,
        use_negative_attention: bool = False,
        mask_root: str | Path | None = None,
        loss_weight: float = 1.0,
        **kwargs: Any,
    ) -> None:
        if _DetectionTrainer is object:
            raise SystemExit("ultralytics is not installed. Install the training extras before running ablations.")
        self.rbccps_use_geometry_attention = use_geometry_attention
        self.rbccps_use_cse = use_cse
        self.rbccps_use_negative_attention = use_negative_attention
        self.rbccps_mask_root = Path(mask_root).expanduser().resolve() if mask_root is not None else None
        if self.rbccps_mask_root is not None and not self.rbccps_mask_root.is_dir():
            raise FileNotFoundError(f"Negative-mask directory does not exist: {self.rbccps_mask_root}")
        self.rbccps_loss_weight = float(loss_weight)
        super().__init__(*args, **kwargs)

    def get_model(self, cfg: Any = None, weights: Any = None, verbose: bool = True) -> Any:
        from ultralytics.nn.tasks import DetectionModel
        from ultralytics.utils import RANK

        model = DetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        apply_ablation_modules(
            model,
            use_geometry_attention=self.rbccps_use_geometry_attention,
            use_cse=self.rbccps_use_cse,
            use_negative_attention=self.rbccps_use_negative_attention,
            negative_mask_loss_weight=self.rbccps_loss_weight,
        )
        # Match Ultralytics DetectionTrainer.get_model, but insert modules before
        # load() so a resumed last.pt restores their learned weights as well.
        if weights:
            model.load(weights)
        return model

    def set_model_attributes(self) -> None:
        super().set_model_attributes()
        if self.rbccps_use_negative_attention:
            install_negative_attention_loss(self.model, loss_weight=self.rbccps_loss_weight)

    def resume_training(self, ckpt: Any) -> None:
        super().resume_training(ckpt)
        # YOLO26 end-to-end resume replaces criterion with init_criterion().
        if self.rbccps_use_negative_attention:
            install_negative_attention_loss(self.model, loss_weight=self.rbccps_loss_weight)

    def get_validator(self) -> Any:
        validator = super().get_validator()
        if self.rbccps_use_negative_attention:
            self.loss_names = "box_loss", "cls_loss", "dfl_loss", "mask_loss"
        return validator

    def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        batch = super().preprocess_batch(batch)
        if self.rbccps_use_negative_attention and "negative_mask" not in batch:
            if self.rbccps_mask_root is None:
                raise ValueError("Negative-attention batch has no negative_mask and no --negative-mask-root was provided.")
            batch["negative_mask"] = _load_batch_negative_masks(batch, self.rbccps_mask_root)
        return batch


def ablation_trainer(
    *,
    use_geometry_attention: bool = False,
    use_cse: bool = False,
    use_negative_attention: bool = False,
    mask_root: str | Path | None = None,
    loss_weight: float = 1.0,
) -> Any:
    if _DetectionTrainer is object:
        raise SystemExit("ultralytics is not installed. Install the training extras before running ablations.")
    return partial(
        RBCCPSAblationTrainer,
        use_geometry_attention=use_geometry_attention,
        use_cse=use_cse,
        use_negative_attention=use_negative_attention,
        mask_root=mask_root,
        loss_weight=loss_weight,
    )


def _c2f_shortcut(module: Any) -> bool:
    first = next(iter(module.m), None)
    return bool(getattr(first, "add", False))


def _c2f_groups(module: Any) -> int:
    first = next(iter(module.m), None)
    conv = getattr(getattr(first, "cv2", None), "conv", None)
    return int(getattr(conv, "groups", 1))


def _c2f_expansion(module: Any) -> float:
    hidden = int(getattr(module, "c", 0))
    output = int(module.cv2.conv.out_channels)
    if not hidden or not output:
        return 0.5
    return hidden / output


def _module_output_channels(module: Any) -> int | None:
    for attr in ("cv3", "cv2"):
        conv = getattr(getattr(module, attr, None), "conv", None)
        channels = getattr(conv, "out_channels", None)
        if channels:
            return int(channels)
    return None


def _detect_heads(pytorch_model: Any) -> list[Any]:
    return [
        module
        for module in pytorch_model.modules()
        if module.__class__.__name__.endswith("Detect") and hasattr(module, "forward")
    ]


def _patch_detect_head(
    detect_head: Any,
    *,
    use_geometry_attention: bool,
    use_negative_attention: bool,
    attention_config: dict[str, Any] | None = None,
) -> None:
    channels = _detect_head_input_channels(detect_head)
    _ensure_attention_modules(
        detect_head,
        channels,
        use_geometry_attention=use_geometry_attention,
        use_negative_attention=use_negative_attention,
        attention_config=attention_config,
    )
    has_hook = _has_attention_input_hook(detect_head)
    if getattr(detect_head, "_rbccps_attention_patched", False) and has_hook:
        detect_head._rbccps_use_geometry_attention = (
            detect_head._rbccps_use_geometry_attention or use_geometry_attention
        )
        detect_head._rbccps_use_negative_attention = (
            detect_head._rbccps_use_negative_attention or use_negative_attention
        )
        return

    detect_head._rbccps_use_geometry_attention = use_geometry_attention
    detect_head._rbccps_use_negative_attention = use_negative_attention
    detect_head._rbccps_negative_mask = None
    if not has_hook:
        detect_head.register_forward_pre_hook(DetectAttentionInputHook())
    detect_head._rbccps_attention_patched = True


def _ensure_attention_modules(
    detect_head: Any,
    channels: tuple[int, ...],
    *,
    use_geometry_attention: bool,
    use_negative_attention: bool,
    attention_config: dict[str, Any] | None = None,
) -> None:
    geometry_config = None
    negative_config = None
    if attention_config is not None:
        geometry_config = attention_config.get("geometry_attention")
        negative_config = attention_config.get("negative_attention")
    if use_geometry_attention and not hasattr(detect_head, "rbccps_geometry_attention"):
        from rbccps_od.models.attention import GeometryAttention

        detect_head.rbccps_geometry_attention = GeometryAttention(channels, config=geometry_config)
    if use_negative_attention and not hasattr(detect_head, "rbccps_negative_attention"):
        from rbccps_od.models.attention import NegativeAttention

        detect_head.rbccps_negative_attention = NegativeAttention(channels, config=negative_config)


def _has_attention_input_hook(detect_head: Any) -> bool:
    return any(isinstance(hook, DetectAttentionInputHook) for hook in detect_head._forward_pre_hooks.values())


def _register_negative_mask_hooks(pytorch_model: Any, detect_heads: list[Any]) -> None:
    if getattr(pytorch_model, "_rbccps_negative_mask_hooks", None):
        return
    pytorch_model.register_forward_pre_hook(NegativeMaskRootPreHook(detect_heads))
    pytorch_model.register_forward_hook(NegativeMaskRootPostHook(detect_heads))
    pytorch_model._rbccps_negative_mask_hooks = True


def install_negative_attention_loss(pytorch_model: Any, *, loss_weight: float = 1.0) -> None:
    """Wrap the model criterion with the negative-mask BCE auxiliary loss."""

    if not _detect_heads(pytorch_model):
        return
    if getattr(pytorch_model, "criterion", None) is None:
        pytorch_model.criterion = pytorch_model.init_criterion()
    if isinstance(pytorch_model.criterion, NegativeAttentionLossWrapper):
        pytorch_model.criterion.loss_weight = float(loss_weight)
        pytorch_model.criterion._model_ref = weakref.ref(pytorch_model)
        return
    pytorch_model.criterion = NegativeAttentionLossWrapper(pytorch_model.criterion, pytorch_model, loss_weight)


def collect_negative_attention_loss(pytorch_model: Any, target_mask: torch.Tensor | None) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for head in _detect_heads(pytorch_model):
        attention = getattr(head, "rbccps_negative_attention", None)
        if attention is not None and getattr(head, "_rbccps_use_negative_attention", False):
            losses.append(attention.mask_loss(target_mask))
    if losses:
        return torch.stack(losses).mean()
    parameter = next(pytorch_model.parameters(), None)
    if parameter is None:
        return torch.tensor(0.0)
    return parameter.sum() * 0.0


def _extract_negative_mask(payload: Any) -> Any:
    if isinstance(payload, dict):
        for key in ("negative_mask", "negative_masks", "seg_mask", "seg_masks"):
            if key in payload:
                return payload[key]
    return None


def _detect_head_input_channels(detect_head: Any) -> tuple[int, ...]:
    channels: list[int] = []
    for scale_head in getattr(detect_head, "cv2", []):
        channel = _first_conv_in_channels(scale_head)
        if channel is None:
            raise ValueError("Could not infer YOLO detect-head input channels for attention modules.")
        channels.append(channel)
    if not channels:
        raise ValueError("YOLO detect head exposes no feature scales for attention modules.")
    return tuple(channels)


def _first_conv_in_channels(module: Any) -> int | None:
    for child in module.modules():
        conv = getattr(child, "conv", None)
        if conv is not None and hasattr(conv, "in_channels"):
            return int(conv.in_channels)
        if hasattr(child, "in_channels"):
            return int(child.in_channels)
    return None


def _load_batch_negative_masks(batch: dict[str, Any], mask_root: Path) -> Any:
    image_tensor = batch["img"]
    image_paths = batch.get("im_file") or batch.get("img_path") or []
    if isinstance(image_paths, (str, Path)):
        image_paths = [image_paths]
    height, width = image_tensor.shape[-2:]
    batch_size = int(image_tensor.shape[0])
    if len(image_paths) != batch_size:
        raise ValueError(
            f"Negative-mask batch has {batch_size} images but {len(image_paths)} image paths; "
            "cannot align supervision masks."
        )
    masks = [
        _load_negative_mask_for_image(Path(image_path), mask_root, size=(height, width))
        for image_path in image_paths
    ]
    return torch.stack(masks, dim=0).to(device=image_tensor.device, dtype=image_tensor.dtype)


def _load_negative_mask_for_image(image_path: Path, mask_root: Path, *, size: tuple[int, int]) -> Any:
    import numpy as np
    import torch.nn.functional as F
    from PIL import Image

    from rbccps_od.models.attention import candidate_negative_mask_paths

    candidates = candidate_negative_mask_paths(image_path, mask_root)
    mask_path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if mask_path is None:
        raise FileNotFoundError(
            f"No negative mask for {image_path}; checked: {', '.join(str(path) for path in candidates)}"
        )

    suffix = mask_path.suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg"}:
        with Image.open(mask_path) as image:
            mask = image.convert("L")
            data = torch.from_numpy(np.array(mask, dtype=np.float32, copy=True)).unsqueeze(0)
    elif suffix == ".npy":
        data = torch.as_tensor(np.load(mask_path), dtype=torch.float32)
    elif suffix == ".pt":
        data = torch.as_tensor(torch.load(mask_path, map_location="cpu"), dtype=torch.float32)
    else:
        raise ValueError(f"Unsupported negative mask format: {mask_path}")

    if data.ndim == 2:
        data = data.unsqueeze(0)
    if data.ndim != 3 or data.shape[0] != 1:
        raise ValueError(f"Negative mask must have one channel and shape HxW or 1xHxW: {mask_path}")
    if float(data.max()) > 1.0:
        data = data / 255.0

    # Ultralytics LetterBox preserves aspect ratio and pads to the batch shape.
    # Apply the same geometry to external masks; a direct square resize shifts labels.
    with Image.open(image_path) as image:
        image_width, image_height = image.size
    height, width = size
    if data.shape[-2:] != (image_height, image_width):
        data = F.interpolate(data.unsqueeze(0), size=(image_height, image_width), mode="nearest").squeeze(0)
    scale = min(height / image_height, width / image_width)
    scaled_width = round(image_width * scale)
    scaled_height = round(image_height * scale)
    data = F.interpolate(data.unsqueeze(0), size=(scaled_height, scaled_width), mode="nearest").squeeze(0)
    left = round((width - scaled_width) / 2 - 0.1)
    top = round((height - scaled_height) / 2 - 0.1)
    right = width - scaled_width - left
    bottom = height - scaled_height - top
    return F.pad(data, (left, right, top, bottom), value=0.0).clamp(0.0, 1.0)
