"""Configurable small CNN classifier for CIFAR-10."""

import torch.nn as nn

from src.config import ModelConfig


class CIFAR10CNN(nn.Module):
    """3 conv blocks (Conv-BN-ReLU x2 + MaxPool) -> classifier head.

    Width and dropout are hyperparameters so HPO can search over them without
    changing the architecture code.
    """

    def __init__(self, cfg: ModelConfig, n_classes: int = 10):
        super().__init__()
        c = cfg.base_channels

        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(3, c),  # 32x32 -> 16x16
            block(c, c * 2),  # 16x16 -> 8x8
            block(c * 2, c * 4),  # 8x8 -> 4x4
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(cfg.dropout),
            nn.Linear(c * 4, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
