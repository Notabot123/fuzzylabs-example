from src.onnx_export import export_to_onnx, verify_onnx
from src.train import save_checkpoint


def test_export_and_verify_onnx_roundtrip(tiny_model, tiny_model_cfg, tmp_path):
    ckpt_path = tmp_path / "model.pt"
    save_checkpoint(tiny_model, tiny_model_cfg, ckpt_path)

    onnx_path = export_to_onnx(ckpt_path, onnx_path=tmp_path / "model.onnx")
    assert onnx_path.exists()

    assert verify_onnx(ckpt_path, onnx_path) is True
