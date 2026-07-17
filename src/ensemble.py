"""Snapshot ensembling via cosine warm restarts.

Rather than training N independent models from scratch (expensive), we take
the already-trained best model and continue training it under a cyclical
(CosineAnnealingWarmRestarts) LR schedule, saving a checkpoint at the end of
each restart cycle. Each snapshot has settled into a different local minimum,
so averaging their softmax outputs behaves like a cheap ensemble.
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import CHECKPOINT_DIR, DEVICE, ModelConfig, TrainConfig
from src.train import evaluate, load_checkpoint, save_checkpoint, train_one_epoch


def train_snapshot_ensemble(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    model_cfg: ModelConfig,
    train_loader,
    val_loader,
    train_cfg: TrainConfig,
    device=DEVICE,
    checkpoint_dir: Path = CHECKPOINT_DIR,
):
    """Continues training `model` and writes one checkpoint per restart cycle.

    Returns the list of snapshot checkpoint paths.
    """
    epochs_per_cycle = max(1, train_cfg.warm_restart_epochs // train_cfg.n_snapshots)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=epochs_per_cycle)
    criterion = nn.CrossEntropyLoss()

    snapshot_paths = []
    for snapshot_idx in range(train_cfg.n_snapshots):
        for _ in range(epochs_per_cycle):
            train_one_epoch(model, train_loader, optimizer, criterion, device)
            scheduler.step()
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        path = checkpoint_dir / f"snapshot_{snapshot_idx}.pt"
        save_checkpoint(model, model_cfg, path, extra={"val_acc": val_acc, "snapshot_idx": snapshot_idx})
        snapshot_paths.append(path)
    return snapshot_paths


def load_ensemble(paths, device=DEVICE):
    return [load_checkpoint(p, device=device)[0] for p in paths]


@torch.no_grad()
def ensemble_predict_proba(models, x: torch.Tensor) -> torch.Tensor:
    """Mean softmax probability across ensemble members. x: (batch, 3, 32, 32)."""
    probs = torch.stack([F.softmax(m(x), dim=1) for m in models], dim=0)
    return probs.mean(dim=0)
