"""Prediction endpoints: single best model, snapshot ensemble, and ONNX."""

import torch
from fastapi import APIRouter, HTTPException, Request

from api.inference import predict_ensemble, predict_onnx, predict_single
from api.schemas import ImageDoc, PredictionDoc
from api.state import ModelRegistry

router = APIRouter(prefix="/predict", tags=["predict"])


def get_registry(request: Request) -> ModelRegistry:
    return request.app.state.registry


def _to_tensor(doc: ImageDoc) -> torch.Tensor:
    return torch.from_numpy(doc.tensor).float().unsqueeze(0)


@router.post("", response_model=PredictionDoc)
def predict(doc: ImageDoc, request: Request):
    registry = get_registry(request)
    if registry.best_model is None:
        raise HTTPException(503, "best model checkpoint not loaded — train a model first")
    return predict_single(registry, _to_tensor(doc))


@router.post("/ensemble", response_model=PredictionDoc)
def predict_ensemble_route(doc: ImageDoc, request: Request):
    registry = get_registry(request)
    if not registry.ensemble_models:
        raise HTTPException(503, "no ensemble checkpoints loaded — run warm-restart training first")
    return predict_ensemble(registry, _to_tensor(doc))


@router.post("/onnx", response_model=PredictionDoc)
def predict_onnx_route(doc: ImageDoc, request: Request):
    registry = get_registry(request)
    if registry.onnx_session is None:
        raise HTTPException(503, "no ONNX model loaded — export a model first")
    return predict_onnx(registry, _to_tensor(doc))
