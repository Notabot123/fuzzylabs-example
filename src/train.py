"""Reusable train/eval loops and checkpoint (de)serialization.

Shared by the HPO search, the final-model training script, warm-restart
snapshot ensembling, and the FastAPI fine-tune endpoint — so there is one
place that knows how to take a step and one place that knows the checkpoint
format.
"""

from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn

from src.config import DEVICE, ModelConfig
from src.models.cnn import CIFAR10CNN


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n += x.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n += x.size(0)
    return total_loss / n, correct / n


def build_model_and_optimizer(model_cfg: ModelConfig, device=DEVICE):
    model = CIFAR10CNN(model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=model_cfg.lr, weight_decay=model_cfg.weight_decay
    )
    return model, optimizer


def train_model(
    model_cfg: ModelConfig,
    train_loader,
    val_loader,
    epochs: int,
    device=DEVICE,
    model=None,
    optimizer=None,
    scheduler=None,
):
    """Train (or continue training, if `model`/`optimizer` are passed in).

    Returns (model, optimizer, history) so callers can chain further training
    (e.g. warm-restart snapshots) without rebuilding the model from scratch.
    """
    if model is None or optimizer is None:
        model, optimizer = build_model_and_optimizer(model_cfg, device)
    criterion = nn.CrossEntropyLoss()

    history = []
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        if scheduler is not None:
            scheduler.step()
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )
    return model, optimizer, history


def save_checkpoint(model: nn.Module, model_cfg: ModelConfig, path, extra: dict | None = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"state_dict": model.state_dict(), "model_cfg": asdict(model_cfg)}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path, device=DEVICE):
    payload = torch.load(path, map_location=device, weights_only=False)
    model_cfg = ModelConfig(**payload["model_cfg"])
    model = CIFAR10CNN(model_cfg).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, model_cfg, payload
