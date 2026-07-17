"""Shared prediction + logging logic used by every predict route.

Every route funnels through here so OOD flagging and monitoring logging
happen exactly once, in one place, regardless of which model served the
request.
"""

import numpy as np
import torch
import torch.nn.functional as F

from src.config import CIFAR10_CLASSES
from src.data import CIFAR10_MEAN, CIFAR10_STD
from src.monitoring.db import log_input_stats, log_prediction
from src.monitoring.drift import compute_image_stats
from src.ood import is_ood

_MEAN = torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1)
_STD = torch.tensor(CIFAR10_STD).view(1, 3, 1, 1)


def normalize_for_classifier(x01: torch.Tensor, device: torch.device) -> torch.Tensor:
    return (x01.to(device) - _MEAN.to(device)) / _STD.to(device)


def _ood_flag(registry, x01: torch.Tensor) -> bool:
    if registry.ood_model is None or registry.ood_threshold is None:
        return False
    return bool(is_ood(registry.ood_model, registry.ood_threshold, x01, registry.device)[0])


def _log(model_name: str, probs: np.ndarray, ood_flag: bool, x01: torch.Tensor):
    predicted_idx = int(probs.argmax())
    log_prediction(
        model_name=model_name,
        predicted_class=CIFAR10_CLASSES[predicted_idx],
        confidence=float(probs[predicted_idx]),
        probs=probs.tolist(),
        is_ood=ood_flag,
    )
    stats = compute_image_stats(x01)
    log_input_stats(**stats)


@torch.no_grad()
def predict_single(registry, x01: torch.Tensor) -> dict:
    """x01: (1, 3, 32, 32) float tensor in [0, 1]."""
    x = normalize_for_classifier(x01, registry.device)
    probs = F.softmax(registry.best_model(x), dim=1)[0].cpu().numpy()
    ood_flag = _ood_flag(registry, x01)
    _log("best_model", probs, ood_flag, x01)
    return _to_response(probs, ood_flag)


@torch.no_grad()
def predict_ensemble(registry, x01: torch.Tensor) -> dict:
    x = normalize_for_classifier(x01, registry.device)
    all_probs = torch.stack([F.softmax(m(x), dim=1) for m in registry.ensemble_models], dim=0)
    probs = all_probs.mean(dim=0)[0].cpu().numpy()
    ood_flag = _ood_flag(registry, x01)
    _log("ensemble", probs, ood_flag, x01)
    return _to_response(probs, ood_flag)


@torch.no_grad()
def predict_onnx(registry, x01: torch.Tensor) -> dict:
    x = normalize_for_classifier(x01, torch.device("cpu")).numpy().astype(np.float32)
    logits = registry.onnx_session.run(None, {"image": x})[0]
    probs = F.softmax(torch.from_numpy(logits), dim=1)[0].numpy()
    ood_flag = _ood_flag(registry, x01)
    _log("onnx", probs, ood_flag, x01)
    return _to_response(probs, ood_flag)


def _to_response(probs: np.ndarray, ood_flag: bool) -> dict:
    predicted_idx = int(probs.argmax())
    return {
        "predicted_class": CIFAR10_CLASSES[predicted_idx],
        "confidence": float(probs[predicted_idx]),
        "probs": probs.astype(np.float32),
        "is_ood": ood_flag,
    }
