"""HAD-MSF reconstruction network.

The graph and fusion implementation follows the computation graph described
in the paper: AdaWAT -> SF-Mamba, spatial and spectral GCNs -> adaptive gate.
"""
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from vmunet.vmamba import VSSLayer


class AdaWAT(nn.Module):
    """Single-level learnable depthwise wavelet-like analysis transform."""

    def __init__(self, channels, kernel_size=3):
        super().__init__()
        if kernel_size % 2 == 0 or kernel_size < 3:
            raise ValueError("kernel_size must be an odd integer >= 3")
        pad = kernel_size // 2

        def depthwise():
            return nn.Conv2d(channels, channels, kernel_size, padding=pad,
                             groups=channels, bias=False)

        self.low_filter = depthwise()
        self.high_filters = nn.ModuleList([depthwise(), depthwise(), depthwise()])
        low = torch.tensor([[0.25, 0.5, 0.25],
                            [0.5, 1.0, 0.5],
                            [0.25, 0.5, 0.25]], dtype=torch.float32)
        low = low / low.sum()
        high_h = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]) / 4.
        high_v = high_h.t()
        high_d = torch.tensor([[1., 0., -1.], [0., 0., 0.], [-1., 0., 1.]]) / 2.

        def embed(kernel):
            if kernel_size == 3:
                return kernel
            expanded = torch.zeros(kernel_size, kernel_size, dtype=kernel.dtype)
            center = kernel_size // 2
            expanded[center - 1:center + 2, center - 1:center + 2] = kernel
            return expanded

        for layer, kernel in zip([self.low_filter, *self.high_filters],
                                 [low, high_h, high_v, high_d]):
            layer.weight.data.copy_(embed(kernel)[None, None].repeat(channels, 1, 1, 1))

    def forward(self, x):
        low = self.low_filter(x)
        highs = tuple(layer(x) for layer in self.high_filters)
        return low, highs


class SpatialFrequencyMamba(nn.Module):
    def __init__(self, channels, depth=1, state_dim=4):
        super().__init__()
        self.vss_layer = VSSLayer(
            dim=channels, depth=depth, d_state=state_dim,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )
        self.high_gates = nn.ModuleList([
            nn.Sequential(nn.Conv2d(channels, channels, 1), nn.Sigmoid())
            for _ in range(3)
        ])
        self.transition = nn.Conv2d(channels, channels, 1)

    def forward(self, low, highs):
        low_features = self.vss_layer(low.permute(0, 2, 3, 1))
        low_features = low_features.permute(0, 3, 1, 2).contiguous()
        high_features = [
            gate(high) * high for gate, high in zip(self.high_gates, highs)
        ]
        return self.transition(low_features + sum(high_features) / 3.0)


class Encoder(nn.Module):
    def __init__(self, shape, adawat_kernel_size=3):
        super().__init__()
        self.adawat = AdaWAT(channels=shape[-1], kernel_size=adawat_kernel_size)

    def forward(self, x):
        return (*self.adawat(x),)


class SpectralGCN(nn.Module):
    """Fixed bidirectional adjacent-band chain, independently per pixel."""

    def __init__(self, bands, hidden_dim=64):
        super().__init__()
        self.bands = bands
        self.conv1 = nn.Linear(1, hidden_dim)
        self.conv2 = nn.Linear(hidden_dim, 1)
        adjacency = torch.zeros(bands, bands)
        diagonal = torch.arange(bands)
        adjacency[diagonal, diagonal] = 1.0
        if bands > 1:
            left = torch.arange(bands - 1)
            adjacency[left, left + 1] = 1.0
            adjacency[left + 1, left] = 1.0
        degree = adjacency.sum(dim=1)
        self.register_buffer("adjacency", adjacency / torch.sqrt(degree[:, None] * degree[None, :]))

    def forward(self, x):
        b, c, h, w = x.shape
        if c != self.bands:
            raise ValueError(f"expected {self.bands} bands, got {c}")
        features = x.permute(0, 2, 3, 1).reshape(-1, c, 1)
        features = torch.matmul(self.adjacency, features)
        features = F.relu(self.conv1(features))
        features = torch.matmul(self.adjacency, features)
        features = self.conv2(features).squeeze(-1)
        return features.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()


class SpatialGCN(nn.Module):
    """Bidirectional four-neighbor pixel graph with self-loops."""

    def __init__(self, in_channels, hidden_dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, 1)
        self.conv2 = nn.Conv2d(hidden_dim, in_channels, 1)

    @staticmethod
    def aggregate(x):
        _, _, h, w = x.shape
        padded = F.pad(x, (1, 1, 1, 1), value=0.0)
        neighbors = (
            padded[:, :, :-2, 1:-1] + padded[:, :, 2:, 1:-1] +
            padded[:, :, 1:-1, :-2] + padded[:, :, 1:-1, 2:]
        )
        degree = torch.ones((1, 1, h, w), device=x.device, dtype=x.dtype)
        degree[:, :, 1:, :] += 1
        degree[:, :, :-1, :] += 1
        degree[:, :, :, 1:] += 1
        degree[:, :, :, :-1] += 1
        return (x + neighbors) / degree

    def forward(self, x):
        x = F.relu(self.conv1(self.aggregate(x)))
        return self.conv2(self.aggregate(x))


class SSGCNBranch(nn.Module):
    def __init__(self, in_channels, hidden_dim=64):
        super().__init__()
        self.spatial_gcn = SpatialGCN(in_channels, hidden_dim)
        self.spectral_gcn = SpectralGCN(in_channels, hidden_dim)

    def forward(self, x):
        return self.spatial_gcn(x) + self.spectral_gcn(x)


class GatingNetwork(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels * 2, channels, 1)

    def forward(self, x, y):
        gate = torch.sigmoid(self.conv(torch.cat([x, y], dim=1)))
        return gate * x + (1.0 - gate) * y


class Decoder(nn.Module):
    def __init__(self, shape, vss_depth=1, state_dim=4):
        super().__init__()
        bands = shape[-1]
        self.sf_mamba = SpatialFrequencyMamba(bands, vss_depth, state_dim)
        self.ssgcn = SSGCNBranch(bands, hidden_dim=64)
        self.gating = GatingNetwork(bands)
        self.reconstruction = nn.Conv2d(bands, bands, 1)

    def forward(self, low, highs, x):
        sf_features = self.sf_mamba(low, highs)
        ss_features = self.ssgcn(x)
        return self.reconstruction(self.gating(sf_features, ss_features))


class RAM(nn.Module):
    def __init__(self, shape, adawat_kernel_size=3, vss_depth=1, state_dim=4):
        super().__init__()
        self.shape = tuple(shape)
        self.name = f"vssdepth{vss_depth}_k{adawat_kernel_size}"
        self.encoder = Encoder(self.shape, adawat_kernel_size)
        self.decoder = Decoder(self.shape, vss_depth, state_dim)

    def forward(self, x):
        return_hwc = x.dim() == 3
        if return_hwc:
            x_bchw = x.permute(2, 0, 1).unsqueeze(0)
        elif x.dim() == 4:
            x_bchw = x
        else:
            raise ValueError("RAM expects HxWxB or BxCxHxW")
        low, highs = self.encoder(x_bchw)
        reconstructed = self.decoder(low, highs, x_bchw)
        if return_hwc:
            return reconstructed.squeeze(0).permute(1, 2, 0).contiguous()
        return reconstructed
