import torch

from scripts.adversarial_demo import fgsm_perturb


def test_fgsm_zero_epsilon_returns_input_unchanged(tiny_model, device):
    x = torch.rand(4, 3, 32, 32)
    y = torch.randint(0, 10, (4,))
    x_adv = fgsm_perturb(tiny_model, x, y, epsilon=0.0, device=device)
    assert torch.allclose(x_adv, x)


def test_fgsm_perturbation_respects_epsilon_budget_and_pixel_range(tiny_model, device):
    x = torch.rand(4, 3, 32, 32)
    y = torch.randint(0, 10, (4,))
    epsilon = 0.05
    x_adv = fgsm_perturb(tiny_model, x, y, epsilon=epsilon, device=device)

    assert x_adv.shape == x.shape
    assert (x_adv >= 0).all() and (x_adv <= 1).all()
    # each pixel moves by at most epsilon (before clipping)
    assert (x_adv - x).abs().max() <= epsilon + 1e-6
    assert not x_adv.requires_grad
