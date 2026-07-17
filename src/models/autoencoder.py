"""Small convolutional autoencoder used for OOD detection via reconstruction error."""

import torch.nn as nn

from src.config import OODConfig


class ConvAutoencoder(nn.Module):
    def __init__(self, cfg: OODConfig):
        super().__init__()
        d = cfg.ae_latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),  # 32x32 -> 16x16
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 16x16 -> 8x8
            nn.ReLU(inplace=True),
            nn.Conv2d(64, d, 3, stride=2, padding=1),  # 8x8 -> 4x4
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(d, 64, 3, stride=2, padding=1, output_padding=1),  # 4x4 -> 8x8
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  # 8x8 -> 16x16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),  # 16x16 -> 32x32
            nn.Sigmoid(),  # output in [0, 1], matches the un-normalized input transform
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))
