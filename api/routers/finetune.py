"""Fine-tune the currently loaded best model on a small labeled batch.

Demo-scope: fine-tunes the in-memory model directly on whatever batch is
POSTed (no held-out validation, no early stopping) and overwrites the best
model checkpoint. It's the building block for closing the OOD feedback loop
(see README) — a human labels a sample of images the monitoring pipeline
flagged as OOD/low-confidence, then POSTs them here.
"""

from dataclasses import replace

import numpy as np
import torch
import torch.nn as nn
from fastapi import APIRouter, HTTPException, Request

from api.inference import normalize_for_classifier
from api.schemas import FinetuneRequest, FinetuneResponse
from api.state import BEST_MODEL_PATH, ModelRegistry
from src.train import save_checkpoint

router = APIRouter(prefix="/finetune", tags=["finetune"])


def get_registry(request: Request) -> ModelRegistry:
    return request.app.state.registry


@router.post("", response_model=FinetuneResponse)
def finetune(body: FinetuneRequest, request: Request):
    registry = get_registry(request)
    if registry.best_model is None or registry.best_model_cfg is None:
        raise HTTPException(503, "best model checkpoint not loaded — train a model first")
    if not body.images:
        raise HTTPException(400, "no images provided")

    x01 = torch.from_numpy(np.stack([img.tensor for img in body.images])).float()
    y = torch.tensor([img.label for img in body.images], dtype=torch.long)
    x = normalize_for_classifier(x01, registry.device)
    y = y.to(registry.device)

    model = registry.best_model
    optimizer = torch.optim.Adam(model.parameters(), lr=body.lr)
    criterion = nn.CrossEntropyLoss()

    history = []
    model.train()
    for epoch in range(body.epochs):
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        acc = (logits.argmax(1) == y).float().mean().item()
        history.append({"epoch": epoch, "loss": loss.item(), "accuracy": acc})
    model.eval()

    registry.best_model_cfg = replace(registry.best_model_cfg, lr=body.lr)
    save_checkpoint(model, registry.best_model_cfg, BEST_MODEL_PATH, extra={"finetuned": True})

    return FinetuneResponse(history=history, checkpoint=str(BEST_MODEL_PATH))
