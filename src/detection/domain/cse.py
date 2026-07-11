import torch
import torch.nn as nn


class CSEBlock(nn.Module):
    """
    Channel Squeeze-and-Excitation Block
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        squeezed_channels = max(1, channels // reduction)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels * 2, squeezed_channels),
            nn.SiLU(),
            nn.Linear(squeezed_channels, channels),
            nn.Sigmoid()
        )

    def forward(self, x):

        b, c, _, _ = x.shape

        avg = self.avg_pool(x).view(b, c)
        mx = self.max_pool(x).view(b, c)

        pooled = torch.cat([avg, mx], dim=1)

        weights = self.fc(pooled).view(b, c, 1, 1)

        return x * weights
