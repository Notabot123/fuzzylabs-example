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

## Project layout

```
src/            training/eval library code (data, models, HPO, ensembling, OOD, monitoring)
api/            FastAPI service (routers, docarray schemas)
tests/          pytest unit tests
scripts/        one-off CLI entry points (train, export, etc.)
configs/        run configuration
checkpoints/    trained model artifacts (gitignored)
.github/        CI workflow
```

## Roadmap / stages

- [x] Project scaffolding
- [ ] Data pipeline + CNN model
- [ ] Training + hyperparameter search
- [ ] Snapshot ensembling + evaluation
- [ ] Out-of-distribution detection
- [ ] Monitoring (confidence + drift logging)
- [ ] ONNX export
- [ ] FastAPI service
- [ ] Tests + CI
- [ ] Design discussion (scaling, drift, adversarial robustness)
