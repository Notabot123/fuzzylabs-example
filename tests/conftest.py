import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch

from src.config import ModelConfig, OODConfig
from src.models.autoencoder import ConvAutoencoder
from src.models.cnn import CIFAR10CNN


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def tiny_model_cfg():
    return ModelConfig(base_channels=4, dropout=0.1, lr=1e-3, weight_decay=0.0)


@pytest.fixture
def tiny_model(tiny_model_cfg, device):
    return CIFAR10CNN(tiny_model_cfg).to(device)


@pytest.fixture
def tiny_ood_cfg():
    return OODConfig(ae_epochs=1, ae_latent_dim=4, threshold_percentile=95.0)


@pytest.fixture
def tiny_autoencoder(tiny_ood_cfg, device):
    return ConvAutoencoder(tiny_ood_cfg).to(device)
