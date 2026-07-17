import torch

from src.ood import is_ood, reconstruction_errors


def test_reconstruction_errors_are_nonnegative(tiny_autoencoder, device):
    from torch.utils.data import DataLoader, TensorDataset

    x = torch.rand(16, 3, 32, 32)
    y = torch.zeros(16, dtype=torch.long)
    loader = DataLoader(TensorDataset(x, y), batch_size=4)

    errors = reconstruction_errors(tiny_autoencoder, loader, device=device)
    assert errors.shape == (16,)
    assert (errors >= 0).all()


def test_is_ood_flags_high_error_above_threshold(tiny_autoencoder, device):
    x = torch.rand(4, 3, 32, 32)
    # threshold of -1 means every non-negative reconstruction error is "OOD"
    flags = is_ood(tiny_autoencoder, threshold=-1.0, x=x, device=device)
    assert flags.shape == (4,)
    assert flags.all()


def test_is_ood_does_not_flag_below_impossibly_high_threshold(tiny_autoencoder, device):
    x = torch.rand(4, 3, 32, 32)
    flags = is_ood(tiny_autoencoder, threshold=1e9, x=x, device=device)
    assert not flags.any()
