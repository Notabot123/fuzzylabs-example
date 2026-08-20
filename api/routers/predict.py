"""Prediction endpoints: single best model, snapshot ensemble, ONNX, and batch."""

import numpy as np
import torch
from fastapi import APIRouter, HTTPException, Request

from api.inference import predict_batch, predict_ensemble, predict_onnx, predict_single
from api.schemas import BatchImageDoc, BatchPredictionDoc, ImageDoc, PredictionDoc
from api.state import ModelRegistry

router = APIRouter(prefix="/predict", tags=["predict"])


def get_registry(request: Request) -> ModelRegistry:
    return request.app.state.registry


def _to_tensor(doc: ImageDoc) -> torch.Tensor:
    return torch.from_numpy(doc.tensor).float().unsqueeze(0)


def _to_batch_tensor(doc: BatchImageDoc) -> torch.Tensor:
    return torch.from_numpy(np.stack([img.tensor for img in doc.images])).float()


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


@router.post("/batch", response_model=BatchPredictionDoc)
def predict_batch_route(doc: BatchImageDoc, request: Request):
    registry = get_registry(request)
    if registry.best_model is None:
        raise HTTPException(503, "best model checkpoint not loaded — train a model first")
    if not doc.images:
        raise HTTPException(422, "images list must not be empty")
    return {"predictions": predict_batch(registry, _to_batch_tensor(doc))}
