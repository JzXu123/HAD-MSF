"""Residual anomaly score."""
import torch


def detect(x, decoder_outputs):
    y = decoder_outputs[0] if isinstance(decoder_outputs, (tuple, list)) else decoder_outputs
    if x.ndim == 3 and y.ndim == 4:
        y = y.squeeze(0).permute(1, 2, 0)
    if x.ndim == 4 and y.ndim == 3:
        y = y.permute(2, 0, 1).unsqueeze(0)
    return (x - y).square().sum(dim=-1 if x.ndim == 3 else 1)
