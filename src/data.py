"""CIFAR-10 loading, train/val/test split, and transforms."""

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.config import DataConfig

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def train_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def get_datasets(cfg: DataConfig):
    """Returns (train, val, test) datasets with a fixed, seeded split.

    train/val come from CIFAR-10's train split (augmented vs. non-augmented
    transforms respectively); test is CIFAR-10's held-out test split.
    """
    full_train_aug = datasets.CIFAR10(
        root=str(cfg.data_dir), train=True, download=True, transform=train_transform()
    )
    full_train_eval = datasets.CIFAR10(
        root=str(cfg.data_dir), train=True, download=True, transform=eval_transform()
    )
    test = datasets.CIFAR10(
        root=str(cfg.data_dir), train=False, download=True, transform=eval_transform()
    )

    n_val = int(len(full_train_aug) * cfg.val_fraction)
    generator = torch.Generator().manual_seed(cfg.seed)
    perm = torch.randperm(len(full_train_aug), generator=generator).tolist()
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    train = Subset(full_train_aug, train_idx)
    val = Subset(full_train_eval, val_idx)
    return train, val, test


def get_dataloaders(cfg: DataConfig):
    train, val, test = get_datasets(cfg)
    train_loader = DataLoader(
        train, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
    )
    val_loader = DataLoader(
        val, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )
    test_loader = DataLoader(
        test, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )
    return train_loader, val_loader, test_loader
