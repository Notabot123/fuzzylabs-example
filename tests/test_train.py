import torch
from torch.utils.data import DataLoader, TensorDataset

from src.train import evaluate, load_checkpoint, save_checkpoint, train_model, train_one_epoch


def _synthetic_loader(n=32, batch_size=8):
    x = torch.randn(n, 3, 32, 32)
    y = torch.randint(0, 10, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


def test_train_one_epoch_reduces_loss_over_many_steps(tiny_model, device):
    import torch.nn as nn

    loader = _synthetic_loader(n=64, batch_size=16)
    optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()

    first_loss, _ = train_one_epoch(tiny_model, loader, optimizer, criterion, device)
    losses = [first_loss]
    for _ in range(14):
        loss, _ = train_one_epoch(tiny_model, loader, optimizer, criterion, device)
        losses.append(loss)

    # small fixed batch -> the model should be able to memorize it; check the
    # trend rather than the last single epoch to avoid step-to-step noise.
    assert min(losses[-3:]) < first_loss


def test_evaluate_returns_loss_and_accuracy_in_valid_range(tiny_model, device):
    import torch.nn as nn

    loader = _synthetic_loader()
    loss, acc = evaluate(tiny_model, loader, nn.CrossEntropyLoss(), device)
    assert loss > 0
    assert 0.0 <= acc <= 1.0


def test_train_model_runs_for_requested_epochs(tiny_model_cfg, device):
    train_loader = _synthetic_loader()
    val_loader = _synthetic_loader()
    _, _, history = train_model(tiny_model_cfg, train_loader, val_loader, epochs=3, device=device)
    assert len(history) == 3
    assert all({"train_loss", "train_acc", "val_loss", "val_acc"} <= h.keys() for h in history)


def test_checkpoint_roundtrip_preserves_weights_and_config(tiny_model, tiny_model_cfg, device, tmp_path):
    path = tmp_path / "ckpt.pt"
    save_checkpoint(tiny_model, tiny_model_cfg, path, extra={"note": "test"})

    loaded_model, loaded_cfg, payload = load_checkpoint(path, device=device)

    assert loaded_cfg == tiny_model_cfg
    assert payload["note"] == "test"
    for p1, p2 in zip(tiny_model.parameters(), loaded_model.parameters()):
        assert torch.allclose(p1, p2)
