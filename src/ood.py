"""Out-of-distribution detection via autoencoder reconstruction error.

Train a small conv autoencoder on in-distribution (CIFAR-10) training images.
At inference time, an image that reconstructs poorly relative to what the
autoencoder has learned from CIFAR-10 is flagged as "unknown" — this is a
simple, cheap complement to softmax confidence (which is often overconfident
on OOD inputs).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

from src.config import CHECKPOINT_DIR, DEVICE, DataConfig, OODConfig
from src.data import get_datasets
from src.models.autoencoder import ConvAutoencoder

AE_TRANSFORM = transforms.ToTensor()  # [0, 1] range, matches the autoencoder's Sigmoid output


@dataclass
class OODArtifacts:
    model: nn.Module
    threshold: float
    percentile: float


def _replace_transform(dataset, transform):
    """Swap the transform on a Subset-wrapped torchvision dataset in place."""
    base = dataset.dataset if hasattr(dataset, "dataset") else dataset
    base.transform = transform
    return dataset


def train_autoencoder(cfg: OODConfig, data_cfg: DataConfig, device=DEVICE) -> ConvAutoencoder:
    train, _, _ = get_datasets(data_cfg)
    train = _replace_transform(train, AE_TRANSFORM)
    loader = torch.utils.data.DataLoader(
        train, batch_size=data_cfg.batch_size, shuffle=True, num_workers=data_cfg.num_workers
    )

    model = ConvAutoencoder(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    model.train()
    for _ in range(cfg.ae_epochs):
        for x, _ in loader:
            x = x.to(device)
            optimizer.zero_grad()
            recon = model(x)
            loss = criterion(recon, x)
            loss.backward()
            optimizer.step()
    model.eval()
    return model


@torch.no_grad()
def reconstruction_errors(model: ConvAutoencoder, loader, device=DEVICE) -> np.ndarray:
    model.eval()
    errors = []
    for x, _ in loader:
        x = x.to(device)
        recon = model(x)
        per_sample = torch.mean((recon - x) ** 2, dim=(1, 2, 3))
        errors.append(per_sample.cpu().numpy())
    return np.concatenate(errors)


def calibrate_threshold(model: ConvAutoencoder, data_cfg: DataConfig, cfg: OODConfig, device=DEVICE) -> float:
    """Threshold = the given percentile of in-distribution reconstruction error.

    Calibrated on held-out validation data (not the AE's own training data) so
    the threshold reflects generalization error, not training-set overfit.
    """
    _, val, _ = get_datasets(data_cfg)
    val = _replace_transform(val, AE_TRANSFORM)
    loader = torch.utils.data.DataLoader(val, batch_size=data_cfg.batch_size, shuffle=False)
    errors = reconstruction_errors(model, loader, device=device)
    return float(np.percentile(errors, cfg.threshold_percentile))


@torch.no_grad()
def is_ood(model: ConvAutoencoder, threshold: float, x: torch.Tensor, device=DEVICE) -> torch.Tensor:
    """x: (batch, 3, 32, 32) in [0, 1] range (i.e. AE_TRANSFORM, not the classifier's normalization).

    Returns a bool tensor of shape (batch,).
    """
    model.eval()
    x = x.to(device)
    recon = model(x)
    per_sample = torch.mean((recon - x) ** 2, dim=(1, 2, 3))
    return per_sample.cpu() > threshold


def save_ood_artifacts(model: ConvAutoencoder, threshold: float, cfg: OODConfig, path: Path = CHECKPOINT_DIR / "ood_autoencoder.pt"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "threshold": threshold, "latent_dim": cfg.ae_latent_dim}, path)


def load_ood_artifacts(path: Path = CHECKPOINT_DIR / "ood_autoencoder.pt", device=DEVICE) -> OODArtifacts:
    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = OODConfig(ae_latent_dim=payload["latent_dim"])
    model = ConvAutoencoder(cfg).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return OODArtifacts(model=model, threshold=payload["threshold"], percentile=cfg.threshold_percentile)
