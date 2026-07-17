import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from api.main import app
from api.state import ModelRegistry
from src.config import CIFAR10_CLASSES, ModelConfig, OODConfig
from src.models.autoencoder import ConvAutoencoder
from src.models.cnn import CIFAR10CNN

BASELINE_STATS = {
    "mean": {"mean": 0.5, "std": 0.1},
    "std": {"mean": 0.2, "std": 0.05},
    "skewness": {"mean": 0.0, "std": 0.2},
    "kurtosis": {"mean": -1.0, "std": 0.3},
}


def _image_payload():
    return {"tensor": np.random.rand(3, 32, 32).tolist()}


@pytest.fixture
def client(tmp_path, monkeypatch):
    import src.monitoring.db as db_module

    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "test.db")
    # avoid the finetune route writing to the repo's real checkpoints/ dir
    monkeypatch.setattr("api.routers.finetune.BEST_MODEL_PATH", tmp_path / "best_model_test.pt")

    with TestClient(app) as c:
        device = torch.device("cpu")
        model_cfg = ModelConfig(base_channels=4, dropout=0.1)
        model = CIFAR10CNN(model_cfg).to(device).eval()
        ood_model = ConvAutoencoder(OODConfig(ae_latent_dim=4)).to(device).eval()

        c.app.state.registry = ModelRegistry(
            device=device,
            best_model=model,
            best_model_cfg=model_cfg,
            ensemble_models=[model],
            onnx_session=None,
            ood_model=ood_model,
            ood_threshold=1e9,  # effectively never flags OOD -> simpler assertions
            baseline_input_stats=BASELINE_STATS,
        )
        yield c


@pytest.fixture
def empty_client(tmp_path, monkeypatch):
    import src.monitoring.db as db_module

    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "empty.db")
    with TestClient(app) as c:
        c.app.state.registry = ModelRegistry(device=torch.device("cpu"))
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_returns_valid_prediction(client):
    r = client.post("/predict", json=_image_payload())
    assert r.status_code == 200
    body = r.json()
    assert body["predicted_class"] in CIFAR10_CLASSES
    assert 0.0 <= body["confidence"] <= 1.0
    assert len(body["probs"]) == 10
    assert abs(sum(body["probs"]) - 1.0) < 1e-4
    assert body["is_ood"] is False


def test_predict_ensemble_returns_valid_prediction(client):
    r = client.post("/predict/ensemble", json=_image_payload())
    assert r.status_code == 200
    body = r.json()
    assert body["predicted_class"] in CIFAR10_CLASSES


def test_predict_onnx_returns_503_when_no_onnx_model(client):
    r = client.post("/predict/onnx", json=_image_payload())
    assert r.status_code == 503


def test_predict_without_model_returns_503(empty_client):
    r = empty_client.post("/predict", json=_image_payload())
    assert r.status_code == 503


def test_predict_ensemble_without_models_returns_503(empty_client):
    r = empty_client.post("/predict/ensemble", json=_image_payload())
    assert r.status_code == 503


def test_monitoring_drift_reflects_logged_predictions(client):
    for _ in range(3):
        client.post("/predict", json=_image_payload())

    r = client.get("/monitoring/drift")
    assert r.status_code == 200
    body = r.json()
    assert body["n_samples"] == 3


def test_monitoring_drift_503_without_baseline(empty_client):
    r = empty_client.get("/monitoring/drift")
    assert r.status_code == 503


def test_finetune_updates_model_and_returns_history(client):
    payload = {
        "images": [
            {"tensor": np.random.rand(3, 32, 32).tolist(), "label": i % 10} for i in range(4)
        ],
        "epochs": 1,
        "lr": 1e-4,
    }
    r = client.post("/finetune", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert len(body["history"]) == 1
    assert "loss" in body["history"][0]
