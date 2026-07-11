import torch
import torch.nn as nn

from rbccps_od.models.attention import GeometryAttention, NegativeAttention
from rbccps_od.models.yolo_ablation import NegativeAttentionLossWrapper


def test_geometry_attention_uses_vertical_and_horizontal_kernels():
    module = GeometryAttention((4, 8), kernel_size=7)
    features = [torch.randn(2, 4, 10, 12), torch.randn(2, 8, 5, 6)]

    outputs = module(features)

    assert [output.shape for output in outputs] == [feature.shape for feature in features]
    assert module.blocks[0].conv_v.kernel_size == (7, 1)
    assert module.blocks[0].conv_h.kernel_size == (1, 7)
    assert module.blocks[1].fuse.in_channels == 16
    assert module.blocks[1].fuse.out_channels == 8


def test_negative_attention_predicts_masks_and_computes_bce_loss():
    module = NegativeAttention((4, 8))
    features = [torch.randn(2, 4, 10, 12), torch.randn(2, 8, 5, 6)]
    target_mask = torch.zeros(2, 1, 40, 48)
    target_mask[:, :, 8:24, 10:30] = 1.0

    outputs = module(features)
    loss = module.mask_loss(target_mask)
    loss.backward()

    assert [output.shape for output in outputs] == [feature.shape for feature in features]
    assert [logit.shape for logit in module.last_logits] == [(2, 1, 10, 12), (2, 1, 5, 6)]
    assert loss.ndim == 0
    assert loss.item() > 0
    assert module.branches[0].conv2.weight.grad is not None


class FakeDetect(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rbccps_negative_attention = NegativeAttention(4)
        self._rbccps_use_negative_attention = True


class FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = FakeDetect()


class DummyCriterion:
    def __call__(self, _preds, _batch):
        return torch.tensor([1.0, 2.0, 3.0]), torch.tensor([0.1, 0.2, 0.3])


def test_negative_attention_loss_wrapper_appends_mask_loss_component():
    model = FakeModel()
    model.head.rbccps_negative_attention(torch.randn(2, 4, 8, 8))
    batch = {
        "img": torch.randn(2, 3, 32, 32),
        "negative_mask": torch.ones(2, 1, 32, 32),
    }
    wrapper = NegativeAttentionLossWrapper(DummyCriterion(), model, loss_weight=0.5)

    loss, loss_items = wrapper({}, batch)

    assert loss.shape == (4,)
    assert loss_items.shape == (4,)
    assert loss[-1].item() > 0
    assert loss_items[-1].item() > 0
