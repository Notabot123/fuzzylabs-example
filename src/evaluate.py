"""Evaluation: accuracy/precision/recall/F1 report + confusion matrix.

Works for a single model or an ensemble by accepting any `predict_proba_fn`
(x -> (batch, n_classes) probabilities), so the same code path evaluates and
compares both.
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix

from src.config import CIFAR10_CLASSES, DEVICE


def single_model_predict_proba_fn(model, device=DEVICE):
    def fn(x):
        return F.softmax(model(x.to(device)), dim=1)

    return fn


def ensemble_predict_proba_fn(models, device=DEVICE):
    def fn(x):
        x = x.to(device)
        probs = torch.stack([F.softmax(m(x), dim=1) for m in models], dim=0)
        return probs.mean(dim=0)

    return fn


@torch.no_grad()
def collect_predictions(predict_proba_fn, loader):
    all_preds, all_targets = [], []
    for x, y in loader:
        probs = predict_proba_fn(x)
        all_preds.append(probs.argmax(1).cpu())
        all_targets.append(y)
    return torch.cat(all_preds).numpy(), torch.cat(all_targets).numpy()


def evaluate_predictions(y_true, y_pred, class_names=CIFAR10_CLASSES):
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)
    return report, cm


def save_report(report: dict, cm: np.ndarray, out_dir: Path, name: str):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{name}_report.json", "w") as f:
        json.dump(report, f, indent=2)
    np.savetxt(out_dir / f"{name}_confusion_matrix.csv", cm, fmt="%d", delimiter=",")


def plot_confusion_matrix(cm: np.ndarray, class_names, out_path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im, ax=ax)
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]), ha="center", va="center", fontsize=7,
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
