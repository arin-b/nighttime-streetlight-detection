from __future__ import annotations

from io import BytesIO

import pytest

torch = pytest.importorskip("torch")
from PIL import Image
from torch import nn

from rbccps_od.models import yolo_ablation


class C2f(nn.Module):
    """Small stand-in with the YOLO C2f output-channel interface."""

    def __init__(self) -> None:
        super().__init__()
        self.cv2 = nn.Module()
        self.cv2.conv = nn.Conv2d(3, 3, kernel_size=1)

    def forward(self, x):
        return self.cv2.conv(x)


def test_cse_keeps_pretrained_block_and_survives_checkpoint_roundtrip():
    model = nn.Sequential(C2f())
    block = model[0]
    pretrained = block.cv2.conv.weight.detach().clone()

    summary = yolo_ablation.apply_ablation_modules(model, use_cse=True)
    assert summary.cse_replacements == 1
    assert model[0] is block
    torch.testing.assert_close(block.cv2.conv.weight, pretrained)
    assert hasattr(block, "rbccps_cse")

    checkpoint = BytesIO()
    torch.save(model, checkpoint)
    checkpoint.seek(0)
    restored = torch.load(checkpoint, weights_only=False)
    assert hasattr(restored[0], "rbccps_cse")
    assert len(restored[0]._forward_hooks) == 1
    assert restored(torch.randn(1, 3, 8, 8)).shape == (1, 3, 8, 8)
    yolo_ablation.apply_ablation_modules(restored, use_cse=True)
    assert len(restored[0]._forward_hooks) == 1


def test_external_negative_mask_matches_letterbox_and_missing_mask_fails(tmp_path):
    image_path = tmp_path / "images" / "train" / "frame.jpg"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (20, 10)).save(image_path)

    mask_root = tmp_path / "masks"
    mask_root.mkdir()
    with pytest.raises(FileNotFoundError, match="No negative mask"):
        yolo_ablation._load_negative_mask_for_image(image_path, mask_root, size=(20, 20))

    mask_path = mask_root / "frame.png"
    mask = Image.new("L", (20, 10), 0)
    mask.putpixel((2, 3), 255)
    mask.save(mask_path)

    aligned = yolo_ablation._load_negative_mask_for_image(image_path, mask_root, size=(20, 20))
    assert aligned.shape == (1, 20, 20)
    assert aligned[0, 8, 2].item() == 1.0
    assert aligned[0, 3, 2].item() == 0.0


def test_trainer_injects_modules_before_loading_resume_weights(monkeypatch):
    pytest.importorskip("ultralytics")
    from ultralytics.nn import tasks

    class FakeDetectionModel(nn.Module):
        def __init__(self, _cfg, nc, ch, verbose):
            super().__init__()
            assert (nc, ch, verbose) == (2, 3, False)
            self.block = C2f()
            self.loaded_with_cse = False

        def load(self, _weights):
            self.loaded_with_cse = hasattr(self.block, "rbccps_cse")

    monkeypatch.setattr(tasks, "DetectionModel", FakeDetectionModel)
    trainer = object.__new__(yolo_ablation.RBCCPSAblationTrainer)
    trainer.data = {"nc": 2, "channels": 3}
    trainer.rbccps_use_geometry_attention = False
    trainer.rbccps_use_cse = True
    trainer.rbccps_use_negative_attention = False
    trainer.rbccps_loss_weight = 1.0

    model = trainer.get_model(weights="resume.pt", cfg="yolo26m.yaml", verbose=False)
    assert model.loaded_with_cse
    assert hasattr(model.block, "rbccps_cse")


def test_audit_loader_keeps_trained_checkpoint_unchanged(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from evaluation.eval_pres.audit_config import DetectorConfig
    from rbccps_od.models import audit_detector

    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    trained = nn.Sequential(C2f())
    yolo_ablation.apply_ablation_modules(trained, use_cse=True)
    before_module = trained[0].rbccps_cse
    before_hooks = len(trained[0]._forward_hooks)
    wrapper = SimpleNamespace(model=trained)
    monkeypatch.setattr(audit_detector, "_import_yolo", lambda: lambda _path: wrapper)

    loaded = audit_detector.load_model(DetectorConfig(model_path=str(checkpoint), use_cse=True))
    assert loaded is wrapper
    assert trained[0].rbccps_cse is before_module
    assert len(trained[0]._forward_hooks) == before_hooks
