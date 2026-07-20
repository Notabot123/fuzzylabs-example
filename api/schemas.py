"""docarray-based request/response schemas for typed tensors over FastAPI.

docarray's `BaseDoc` is a pydantic v2 model, so these can be used directly as
FastAPI request bodies / response models while still getting typed, shape-
validated numpy arrays instead of hand-rolled list[list[list[float]]] fields.

Contract: `tensor` is a raw CIFAR-10-sized image, channel-first, float32,
pixel values in [0, 1] (i.e. `ToTensor()` output, *not* normalized). API
internals apply CIFAR-10 normalization for the classifier and use the raw
[0, 1] tensor directly for OOD reconstruction error.
"""

from docarray import BaseDoc, DocList
from docarray.typing import NdArray


class ImageDoc(BaseDoc):
    tensor: NdArray[3, 32, 32]


class BatchImageDoc(BaseDoc):
    images: DocList[ImageDoc]


class PredictionDoc(BaseDoc):
    predicted_class: str
    confidence: float
    probs: list[float]
    is_ood: bool


class BatchPredictionDoc(BaseDoc):
    predictions: DocList[PredictionDoc]


class LabeledImageDoc(BaseDoc):
    tensor: NdArray[3, 32, 32]
    label: int  # index into CIFAR10_CLASSES


class FinetuneRequest(BaseDoc):
    images: DocList[LabeledImageDoc]
    epochs: int = 1
    lr: float = 1e-4


class FinetuneResponse(BaseDoc):
    history: list  # per-epoch {loss, accuracy} on the fine-tune batch
    checkpoint: str
