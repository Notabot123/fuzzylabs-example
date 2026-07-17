"""Export a trained CNN checkpoint to ONNX, and verify it against the PyTorch model."""

from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from src.config import CHECKPOINT_DIR
from src.train import load_checkpoint


def export_to_onnx(checkpoint_path: Path, onnx_path: Path = CHECKPOINT_DIR / "model.onnx", opset: int = 17):
    model, model_cfg, _ = load_checkpoint(checkpoint_path, device=torch.device("cpu"))
    model.eval()

    dummy_input = torch.randn(1, 3, 32, 32)
    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
    )
    return onnx_path


def verify_onnx(checkpoint_path: Path, onnx_path: Path, atol: float = 1e-4) -> bool:
    """Sanity check: PyTorch and ONNXRuntime should agree on the same random input."""
    model, _, _ = load_checkpoint(checkpoint_path, device=torch.device("cpu"))
    model.eval()

    x = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        torch_out = model(x).numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = session.run(None, {"image": x.numpy()})[0]

    return bool(np.allclose(torch_out, onnx_out, atol=atol))


def load_onnx_session(onnx_path: Path = CHECKPOINT_DIR / "model.onnx") -> ort.InferenceSession:
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"]
    return ort.InferenceSession(str(onnx_path), providers=providers)


def onnx_predict(session: ort.InferenceSession, x: np.ndarray) -> np.ndarray:
    """x: (batch, 3, 32, 32) float32 numpy array. Returns raw logits."""
    return session.run(None, {"image": x.astype(np.float32)})[0]
