"""Minimal example: hit /predict and /predict/batch with random tensors via docarray + httpx."""

import numpy as np
import httpx

from api.schemas import ImageDoc, BatchImageDoc

doc = ImageDoc(tensor=np.random.rand(3, 32, 32).astype("float32"))

resp = httpx.post(
    "http://127.0.0.1:8000/predict",
    content=doc.model_dump_json(),
    headers={"Content-Type": "application/json"},
)
resp.raise_for_status()
print(resp.json())


## minibatch: a DocList of images, not a single array with a batch dim -
## NdArray shapes are fixed-size per dimension, so a variable-length batch
## has to be a list of individually-shaped docs instead.
batch_doc = BatchImageDoc(
    images=[ImageDoc(tensor=np.random.rand(3, 32, 32).astype("float32")) for _ in range(16)]
)
resp = httpx.post(
    "http://127.0.0.1:8000/predict/batch",
    content=batch_doc.model_dump_json(),
    headers={"Content-Type": "application/json"},
)
resp.raise_for_status()
print(resp.json())
