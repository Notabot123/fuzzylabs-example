"""Model registry: everything the API needs is loaded once at startup and
stashed here, rather than re-reading checkpoints from disk per request.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import onnxruntime as ort
import torch
import torch.nn as nn

from src.config import CHECKPOINT_DIR, DEVICE, ModelConfig
from src.models.autoencoder import ConvAutoencoder
from src.ood import load_ood_artifacts
from src.train import load_checkpoint

BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pt"
ENSEMBLE_GLOB = "snapshot_*.pt"
ONNX_PATH = CHECKPOINT_DIR / "model.onnx"
BASELINE_STATS_PATH = CHECKPOINT_DIR / "baseline_input_stats.json"


@dataclass
class ModelRegistry:
    device: torch.device
    best_model: nn.Module | None = None
    best_model_cfg: ModelConfig | None = None
    ensemble_models: list = field(default_factory=list)
    onnx_session: ort.InferenceSession | None = None
    ood_model: nn.Module | None = None
    ood_threshold: float | None = None
    baseline_input_stats: dict | None = None

    @classmethod
    def load(cls, checkpoint_dir: Path = CHECKPOINT_DIR, device: torch.device = DEVICE) -> "ModelRegistry":
        registry = cls(device=device)

        if BEST_MODEL_PATH.exists():
            registry.best_model, registry.best_model_cfg, _ = load_checkpoint(BEST_MODEL_PATH, device=device)

        registry.ensemble_models = [
            load_checkpoint(p, device=device)[0]
            for p in sorted(checkpoint_dir.glob(ENSEMBLE_GLOB))
        ]

        if ONNX_PATH.exists():
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if "CUDAExecutionProvider" in ort.get_available_providers()
                else ["CPUExecutionProvider"]
            )
            registry.onnx_session = ort.InferenceSession(str(ONNX_PATH), providers=providers)

        ood_path = checkpoint_dir / "ood_autoencoder.pt"
        if ood_path.exists():
            artifacts = load_ood_artifacts(ood_path, device=device)
            registry.ood_model, registry.ood_threshold = artifacts.model, artifacts.threshold

        if BASELINE_STATS_PATH.exists():
            registry.baseline_input_stats = json.loads(BASELINE_STATS_PATH.read_text())

        return registry
