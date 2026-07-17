"""Central configuration for the CIFAR-10 pipeline.

Defaults are deliberately demo-scale (few epochs, few HPO trials) so the
whole pipeline runs end-to-end in minutes on a single GPU/CPU. Bump the
values below (or override via CLI flags where exposed) for a "real" run.
"""

from dataclasses import dataclass, field
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
DB_PATH = ROOT_DIR / "checkpoints" / "monitoring.db"

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class DataConfig:
    data_dir: Path = DATA_DIR
    val_fraction: float = 0.1  # held out from the 50k training images
    batch_size: int = 128
    num_workers: int = 2
    seed: int = 42


@dataclass
class ModelConfig:
    """Hyperparameters searched by HPO. Defaults are the search midpoint."""

    base_channels: int = 32
    dropout: float = 0.3
    lr: float = 1e-3
    weight_decay: float = 1e-4


@dataclass
class HPOConfig:
    n_trials: int = 6  # demo-scale; raise for a real search
    epochs_per_trial: int = 3
    train_subset_size: int = 8000  # subset of train split used per trial
    lr_range: tuple = (1e-4, 3e-3)
    dropout_range: tuple = (0.1, 0.5)
    base_channels_choices: tuple = (16, 32, 48)
    weight_decay_range: tuple = (1e-6, 1e-3)


@dataclass
class TrainConfig:
    final_epochs: int = 8  # demo-scale full training of the best config
    warm_restart_epochs: int = 4  # extra epochs to harvest ensemble snapshots
    n_snapshots: int = 3  # number of warm-restart checkpoints for the ensemble


@dataclass
class OODConfig:
    ae_epochs: int = 6
    ae_latent_dim: int = 64
    threshold_percentile: float = 95.0  # reconstruction-error percentile cutoff


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    hpo: HPOConfig = field(default_factory=HPOConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    ood: OODConfig = field(default_factory=OODConfig)
