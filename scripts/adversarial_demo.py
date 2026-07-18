"""Demonstrate (and measure) adversarial fragility with an untargeted FGSM attack.

FGSM (Goodfellow et al., 2015) takes one gradient step of size epsilon in the
direction that *increases* the loss for the true label:

    x_adv = clip(x + epsilon * sign(grad_x loss(model(x), y)), 0, 1)

This is a white-box, single-step attack — a weak baseline, not a security
audit — but it's cheap and enough to demonstrate that a plain CNN with no
adversarial training degrades sharply under even small, human-imperceptible
perturbations. Also checks whether the OOD autoencoder (trained only on clean
images) flags adversarial examples at a higher rate — a plausible cheap
signal, though not a robust defense (see README).

Run with: python -m scripts.adversarial_demo
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.inference import normalize_for_classifier
from src.config import CHECKPOINT_DIR, DEVICE, ROOT_DIR, Config
from src.ood import is_ood, load_ood_artifacts
from src.train import load_checkpoint

REPORTS_DIR = ROOT_DIR / "reports"
EPSILONS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]
N_SAMPLES = 500


def fgsm_perturb(model, x01: torch.Tensor, y: torch.Tensor, epsilon: float, device) -> torch.Tensor:
    x01 = x01.clone().to(device).requires_grad_(True)
    logits = model(normalize_for_classifier(x01, device))
    loss = F.cross_entropy(logits, y.to(device))
    model.zero_grad(set_to_none=True)
    loss.backward()

    if epsilon == 0.0:
        return x01.detach()
    perturbation = epsilon * x01.grad.sign()
    return (x01 + perturbation).clamp(0, 1).detach()


def main():
    cfg = Config()
    print(f"device: {DEVICE}")

    model, _, _ = load_checkpoint(CHECKPOINT_DIR / "best_model.pt", device=DEVICE)
    ood = load_ood_artifacts(CHECKPOINT_DIR / "ood_autoencoder.pt", device=DEVICE)

    # Raw [0, 1] pixels: FGSM's perturbation budget is defined in pixel space,
    # and fgsm_perturb re-normalizes internally before the forward pass.
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms

    raw_test = datasets.CIFAR10(
        root=str(cfg.data.data_dir), train=False, download=True, transform=transforms.ToTensor()
    )
    loader = DataLoader(Subset(raw_test, list(range(N_SAMPLES))), batch_size=100)

    results = []
    for epsilon in EPSILONS:
        correct, total, ood_flags = 0, 0, 0
        for x01, y in loader:
            x_adv = fgsm_perturb(model, x01, y, epsilon, DEVICE)
            with torch.no_grad():
                logits = model(normalize_for_classifier(x_adv, DEVICE))
                pred = logits.argmax(1).cpu()
                correct += (pred == y).sum().item()
                total += y.size(0)
                ood_flags += is_ood(ood.model, ood.threshold, x_adv, DEVICE).sum().item()

        acc = correct / total
        ood_rate = ood_flags / total
        results.append({"epsilon": epsilon, "accuracy": acc, "ood_flag_rate": ood_rate})
        print(f"epsilon={epsilon:.2f}  accuracy={acc:.4f}  ood_flag_rate={ood_rate:.4f}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "adversarial_robustness.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"saved {out_path}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(6, 4))
    eps = [r["epsilon"] for r in results]
    ax1.plot(eps, [r["accuracy"] for r in results], "o-", color="tab:blue", label="accuracy")
    ax1.set_xlabel("FGSM epsilon")
    ax1.set_ylabel("accuracy", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(eps, [r["ood_flag_rate"] for r in results], "s--", color="tab:red", label="OOD flag rate")
    ax2.set_ylabel("OOD flag rate", color="tab:red")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "adversarial_robustness.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
