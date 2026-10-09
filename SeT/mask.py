"""Soft min-max anomaly mask used by the label-free SeT+ objective."""
import torch
from typing import Union


class Mask:
    def __init__(self, init: Union[tuple, torch.Tensor], device=None):
        if isinstance(init, tuple):
            self.mask = torch.zeros((*init, 1), device=device, dtype=torch.float32)
        elif isinstance(init, torch.Tensor):
            self.mask = init.to(device=device) if device is not None else init
            if self.mask.ndim == 2:
                self.mask = self.mask.unsqueeze(-1)
        else:
            raise TypeError("init must be a shape tuple or torch.Tensor")

    def dot_prod(self, x):
        return self.mask.to(dtype=x.dtype, device=x.device) * x

    def count(self):
        return self.mask.sum()

    def not_op(self):
        return Mask(1.0 - self.mask, device=self.mask.device)

    def update(self, dm: torch.Tensor):
        dm = dm.detach()
        lower, upper = dm.amin(), dm.amax()
        self.mask = ((dm - lower) / (upper - lower).clamp_min(1e-12)).unsqueeze(-1)
