# CIFAR-10 Classifier + Serving API

A CNN-based CIFAR-10 classifier packaged as a FastAPI service, built as a
demonstration of ML engineering practice rather than an attempt to push
state-of-the-art accuracy on a solved benchmark.

> Status: work in progress — this README is filled in incrementally as each
> stage of the pipeline lands. See the bottom of this file for the roadmap.

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

`requirements.txt` installs a CPU build of PyTorch by default. If you have an
NVIDIA GPU and want CUDA acceleration (recommended — the training pipeline is
~10x faster), install the matching CUDA wheels *before* `pip install -r
requirements.txt` so it's satisfied and left alone:

```bash
pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

(Swap `cu128` for whatever CUDA build matches your driver — see
[pytorch.org/get-started](https://pytorch.org/get-started/locally/).)

## Running it

```bash
# 1. Run the full pipeline: HPO -> final training -> snapshot ensemble ->
#    OOD autoencoder -> ONNX export -> evaluation report -> drift baseline.
#    Downloads CIFAR-10 on first run. Demo-scale by default (~5-10 min on a
#    laptop GPU) — see src/config.py to scale up.
python -m scripts.train_pipeline

# 2. Serve the trained artifacts.
uvicorn api.main:app --reload

# 3. Run the test suite (uses tiny synthetic models/data — no training required).
pytest
```

With the API running, see `/docs` for interactive Swagger UI. Endpoints:

| Endpoint              | Method | Description                                   |
|------------------------|--------|------------------------------------------------|
| `/predict`              | POST   | Single best-model inference                    |
| `/predict/ensemble`     | POST   | Snapshot-ensemble inference (averaged softmax)  |
| `/predict/onnx`         | POST   | ONNX Runtime inference                          |
| `/finetune`              | POST   | Fine-tune the best model on a labeled batch     |
| `/monitoring/drift`      | GET    | Drift report vs. the calibrated baseline        |
| `/health`                | GET    | Liveness check                                  |

## Project layout

```
src/            training/eval library code (data, models, HPO, ensembling, OOD, monitoring)
api/            FastAPI service (routers, docarray schemas)
scripts/        pipeline entry points (train_pipeline.py)
tests/          pytest unit tests (synthetic data/models — no training needed)
checkpoints/    trained model artifacts (gitignored)
reports/        evaluation reports + confusion matrices (committed)
.github/        CI workflow
```

## Roadmap / stages

- [x] Project scaffolding
- [x] Data pipeline + CNN model
- [x] Training + hyperparameter search
- [x] Snapshot ensembling + evaluation
- [x] Out-of-distribution detection
- [x] Monitoring (confidence + drift logging)
- [x] ONNX export
- [x] FastAPI service
- [x] Tests + CI
- [ ] Design discussion (scaling, drift, adversarial robustness)
