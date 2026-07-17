"""FastAPI app assembly: loads model artifacts once at startup, wires up routers."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.routers import finetune, monitoring, predict
from api.state import ModelRegistry
from src.config import DEVICE
from src.monitoring.db import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    app.state.registry = ModelRegistry.load(device=DEVICE)
    yield


app = FastAPI(title="CIFAR-10 Classifier API", lifespan=lifespan)

app.include_router(predict.router)
app.include_router(finetune.router)
app.include_router(monitoring.router)


@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE)}
