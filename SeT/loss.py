"""SeT+ and MSR losses from the paper, implemented on HWC cubes."""
import torch
from torch import nn
import torch.nn.functional as F


def _as_hwc(x):
    if x.ndim == 3:
        return x
    if x.ndim == 4 and x.shape[0] == 1:
        return x[0].permute(1, 2, 0)
    raise ValueError("expected HxWxB or 1xBxHxW")


class LoG(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        kernel = torch.tensor(
            [[-2., -4., -4., -4., -2.],
             [-4., 0., 8., 0., -4.],
             [-4., 8., 24., 8., -4.],
             [-4., 0., 8., 0., -4.],
             [-2., -4., -4., -4., -2.]]
        )
        self.register_buffer("kernel", kernel[None, None])
        self.pad = kernel.shape[0] // 2

    def forward(self, x):
        was_hwc = x.ndim == 3
        hwc = _as_hwc(x)
        h, w, bands = hwc.shape
        bchw = hwc.permute(2, 0, 1).unsqueeze(0)
        padded = F.pad(bchw, (self.pad,) * 4, mode="reflect")
        kernel = self.kernel.to(dtype=bchw.dtype).repeat(bands, 1, 1, 1)
        filtered = F.conv2d(padded, kernel, groups=bands)
        result = filtered.squeeze(0).permute(1, 2, 0).contiguous()
        return result if was_hwc else result.permute(2, 0, 1).unsqueeze(0)


class SeTLoss(nn.Module):
    def __init__(self, lmda=1e-3, device=None, eps=1e-6):
        super().__init__()
        self.lmda = float(lmda)
        self.eps = float(eps)
        self.log = LoG(device)

    @staticmethod
    def _mask(mask):
        return mask.mask if hasattr(mask, "mask") else mask

    def forward(self, **kwargs):
        mask = self._mask(kwargs["mask"]).to(dtype=kwargs["x"].dtype)
        x = _as_hwc(kwargs["x"])
        y = _as_hwc(kwargs["y"])
        if mask.ndim == 2:
            mask = mask.unsqueeze(-1)
        log_y = self.log(y)
        anomaly = mask.expand_as(log_y)
        background = 1.0 - mask
        as_loss = (anomaly * log_y).square().sum() / (mask.sum() + self.eps)
        br_loss = (background * (x - y)).square().sum() / (background.sum() + self.eps)
        return br_loss + self.lmda * as_loss


class MSRLoss(nn.Module):
    """Multi-scale reconstruction loss; the public model has one scale."""
    def forward(self, **kwargs):
        x = _as_hwc(kwargs["x"])
        output = kwargs["y"]
        outputs = output if isinstance(output, (tuple, list)) else (output,)
        total = x.new_zeros(())
        for item in outputs:
            y = _as_hwc(item)
            if y.shape[:2] != x.shape[:2]:
                y_bchw = y.permute(2, 0, 1).unsqueeze(0)
                y = F.interpolate(y_bchw, size=x.shape[:2], mode="bilinear", align_corners=False)
                y = y.squeeze(0).permute(1, 2, 0)
            total = total + (x - y).square().sum()
        return total / len(outputs)


class TotalLoss(nn.Module):
    def __init__(self, lmda=1e-3, device=None, eps=1e-6):
        super().__init__()
        self.set_loss = SeTLoss(lmda, device, eps)
        self.msr_loss = MSRLoss()
        self.eps = float(eps)

    def forward(self, **kwargs):
        bands = _as_hwc(kwargs["x"]).shape[-1]
        return (self.set_loss(**kwargs) + self.msr_loss(**kwargs)) / max(bands, 1)
