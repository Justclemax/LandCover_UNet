"""U-Net (Ronneberger et al., 2015) for per-pixel land cover classification.

Same shape as the reference Keras model (4 levels down, 4 up, skip connections), with
batch normalisation and the input normalisation built into the network, so the exported
ONNX model takes raw physical values (reflectance and dB) and the Rust server needs no
preprocessing beyond filling missing pixels.
"""

import torch
from torch import nn


class DoubleConv(nn.Sequential):
    def __init__(self, cin: int, cout: int):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )


class UNet(nn.Module):
    def __init__(self, in_channels: int, n_classes: int, base: int = 24, dropout: float = 0.1,
                 mean: torch.Tensor | None = None, std: torch.Tensor | None = None):
        super().__init__()
        f = [base * 2**i for i in range(5)]
        self.register_buffer("mean", (mean if mean is not None else torch.zeros(in_channels)).view(1, -1, 1, 1))
        self.register_buffer("std", (std if std is not None else torch.ones(in_channels)).view(1, -1, 1, 1))
        self.down = nn.ModuleList([DoubleConv(in_channels, f[0])] + [DoubleConv(f[i], f[i + 1]) for i in range(3)])
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(DoubleConv(f[3], f[4]), nn.Dropout2d(dropout))
        self.up = nn.ModuleList([nn.ConvTranspose2d(f[i + 1], f[i], 2, stride=2) for i in reversed(range(4))])
        self.dec = nn.ModuleList([DoubleConv(f[i] * 2, f[i]) for i in reversed(range(4))])
        self.head = nn.Conv2d(f[0], n_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(N, C, H, W) raw values -> (N, K, H, W) logits. H and W must be multiples of 16."""
        x = (x - self.mean) / self.std
        skips = []
        for i, block in enumerate(self.down):
            x = block(x if i == 0 else self.pool(x))
            skips.append(x)
        x = self.bottleneck(self.pool(x))
        for up, dec, skip in zip(self.up, self.dec, reversed(skips), strict=True):
            x = dec(torch.cat([up(x), skip], dim=1))
        return self.head(x)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
