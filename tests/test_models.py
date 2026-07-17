import torch


def test_cnn_output_shape(tiny_model):
    x = torch.randn(4, 3, 32, 32)
    logits = tiny_model(x)
    assert logits.shape == (4, 10)


def test_cnn_dropout_and_channels_are_configurable(tiny_model_cfg, device):
    from src.models.cnn import CIFAR10CNN

    model = CIFAR10CNN(tiny_model_cfg).to(device)
    x = torch.randn(2, 3, 32, 32)
    assert model(x).shape == (2, 10)


def test_autoencoder_reconstructs_same_shape(tiny_autoencoder):
    x = torch.rand(4, 3, 32, 32)
    recon = tiny_autoencoder(x)
    assert recon.shape == x.shape
    assert torch.all((recon >= 0) & (recon <= 1))  # sigmoid output
