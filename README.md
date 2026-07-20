# CIFAR-10 Classifier + Serving API

A CNN-based CIFAR-10 classifier packaged as a FastAPI service, built as a
demonstration of ML engineering practice rather than an attempt to push
state-of-the-art accuracy on a solved benchmark.

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
scripts/        pipeline entry points (train_pipeline.py, adversarial_demo.py)
tests/          pytest unit tests (synthetic data/models — no training needed)
checkpoints/    trained model artifacts (gitignored)
reports/        evaluation reports + confusion matrices (committed)
.github/        CI workflow
```

## Results

One demo-scale run (`python -m scripts.train_pipeline`, ~5 min on an RTX 3060
laptop GPU; see `src/config.py` for the knobs to scale this up):

| Stage                          | Result                                              |
|---------------------------------|------------------------------------------------------|
| HPO (6 trials, Optuna TPE)       | best val_acc=0.493 → `base_channels=48, dropout=0.156, lr=2.7e-4` |
| Final training (8 epochs)        | val_acc=0.785                                        |
| **Best single model** (test set) | **accuracy=0.764**, macro F1=0.765                    |
| **Snapshot ensemble** (3 warm restarts, test set) | **accuracy=0.823**, macro F1=0.823 (**+5.9pt over the single model** for ~3x inference cost) |
| OOD autoencoder threshold        | p95 reconstruction error = 0.0072                     |
| ONNX export                      | verified numerically identical to PyTorch output      |

Full per-class precision/recall/F1 and confusion matrices:
[`reports/best_model_report.json`](reports/best_model_report.json) /
[`reports/best_model_confusion_matrix.png`](reports/best_model_confusion_matrix.png),
[`reports/ensemble_report.json`](reports/ensemble_report.json) /
[`reports/ensemble_confusion_matrix.png`](reports/ensemble_confusion_matrix.png).
Cat/dog are consistently the weakest classes (recall 0.63/0.73 for the
ensemble) — the classic CIFAR-10 confusion pair.

**Adversarial robustness** (`python -m scripts.adversarial_demo`, FGSM on 500
test images — see [`reports/adversarial_robustness.json`](reports/adversarial_robustness.json) / `.png`):

| FGSM epsilon | Accuracy | OOD flag rate |
|---|---|---|
| 0.00 (clean) | 0.778 | 0.062 |
| 0.01 | 0.136 | 0.064 |
| 0.02 | 0.032 | 0.072 |
| 0.05 | 0.018 | 0.206 |
| 0.10 | 0.056 | 0.994 |
| 0.20 | 0.074 | 1.000 |

A perturbation budget of just 0.01 (imperceptible — pixel values are in
[0, 1]) collapses accuracy from 78% to 14%, confirming the model has no
adversarial robustness. Notably, the OOD flag rate barely moves at that same
epsilon (0.062 → 0.064) — the reconstruction-error detector only starts
reacting at much larger, more visible perturbations (eps ≥ 0.05). That's
exactly the failure mode described below: the detector doesn't catch the
epsilon range where the attack is actually most effective and stealthy.

## Design discussion

Answers to the brief's five questions, tied to what's actually implemented
(not just described) in this repo.

### 1. Out-of-distribution ("unknown") images

Softmax confidence alone is a poor OOD signal — CNNs are frequently
*confidently* wrong on inputs unlike anything they were trained on. This repo
adds a second, independent signal: [`src/ood.py`](src/ood.py) trains a small
convolutional autoencoder on CIFAR-10 only, and calibrates a reconstruction-
error threshold at the 95th percentile of held-out validation error. Every
`/predict*` response includes an `is_ood` flag computed from this threshold,
alongside the classifier's own softmax confidence.

Limitations of the current approach: a single global MSE threshold is coarse
(no per-class calibration), and autoencoders can generalize reconstruction to
OOD inputs that are visually similar to the training distribution (giving
false negatives). To improve it: (a) fuse multiple signals — reconstruction
error, softmax entropy, and ensemble disagreement (variance across the
snapshot ensemble's predictions) — into a single calibrated score, e.g. via a
small logistic regression fit on a held-out ID/OOD proxy set (CIFAR-100 or
SVHN make good proxies); (b) report AUROC/AUPR against that proxy set rather
than picking a threshold by percentile alone; (c) an energy-based or ODIN-style
score, which the literature generally finds outperforms raw softmax or MSE.

### 2. Scaling the endpoint

The current service loads models once at startup ([`api/state.py`](api/state.py))
and serves requests through synchronous route handlers, which FastAPI already
runs in a threadpool — fine for a demo, not for high throughput, because each
request does its own single-image forward pass.

To scale:
- **Dynamic batching**: queue incoming single-image requests for a short
  window (a few ms, or until N accumulate) and run one batched forward pass —
  GPUs are throughput-bound, not latency-bound, per request. A hand-rolled
  asyncio queue + background worker loop is enough at moderate scale; at
  larger scale, NVIDIA Triton or Ray Serve provide this out of the box along
  with dynamic batching, model warm-up, and multi-model routing.
- **Async work queue**: for latency-tolerant bulk traffic, push requests to a
  queue (SQS/Kafka/Redis Streams) and have a separate autoscaled pool of
  GPU-bound consumers batch-drain and infer, writing results back or invoking
  a callback — decouples request ingestion from GPU capacity.
- **Horizontal scaling**: stateless replicas behind a load balancer, scaled on
  GPU utilization/queue depth. The one piece of state that *doesn't*
  horizontally scale as-is is the SQLite monitoring DB (single-writer,
  single-file) — a real deployment needs to swap that for Postgres/Timescale
  or a managed metrics pipeline before running more than one replica.
- **Faster inference**: the ONNX export ([`src/onnx_export.py`](src/onnx_export.py),
  `/predict/onnx`) is a first step toward TensorRT/quantized serving for lower
  per-request latency.

### 3. Feeding "unknown" volume back into the model

Every inference is logged with its `is_ood` flag
([`src/monitoring/db.py`](src/monitoring/db.py)), and `/monitoring/drift`
([`src/monitoring/drift.py`](src/monitoring/drift.py)) already flags when the
OOD rate over a rolling window crosses 20%. The feedback loop this enables:

1. **Capture** — sample flagged-OOD inputs into a review queue. (Today we only
   log stats, not the raw image; a production version would store the image
   itself, e.g. an S3 pointer alongside the `predictions` row.)
2. **Triage** — cluster the flagged images (e.g. k-means/HDBSCAN on the
   autoencoder's latent vectors) to tell "a new coherent class showing up
   repeatedly" apart from "scattered noise/corrupted inputs" — prioritize
   labeling effort (active learning) on the highest-reconstruction-error,
   most-frequent cluster.
3. **Label** a sample of the new cluster, and either fold it in as a new class
   (retrain with an expanded label set) or confirm it's legitimately
   out-of-scope (tighten monitoring, no model change).
4. **Retrain/fine-tune** — `/finetune` (`api/routers/finetune.py`) is the
   building block: POST labeled images, it fine-tunes the in-memory best
   model and checkpoints the result. As implemented, it's a demo (no
   validation gate, overwrites the serving checkpoint immediately) — a real
   version needs a held-out shadow evaluation before promoting a fine-tuned
   checkpoint, plus checkpoint versioning so a regression can be rolled back.
5. **Recalibrate** the OOD threshold periodically too, since a legitimately
   evolving input distribution will otherwise keep tripping the OOD flag even
   after the model has been updated to handle it.

### 4. Detecting degradation, drift, and rising error rates

Ground-truth labels aren't available at serving time (that's the point of the
model), so everything here is a proxy, logged per-request and summarized by
`drift_summary()`:

- **Confidence drift** — rolling mean softmax confidence per model. A model
  can stay *confidently* wrong, though, so this alone is not sufficient.
- **Input drift** — per-image pixel statistics (mean/std/skewness/kurtosis)
  compared via z-score against a baseline calibrated from the training set
  (`compute_baseline_input_stats`, run once by `scripts/train_pipeline.py`).
  This is deliberately cheap, but also crude — it can miss semantic shifts
  that don't move raw pixel moments. A stronger version compares
  penultimate-layer embedding distributions (Population Stability Index,
  KS-test, or MMD) rather than raw pixels.
- **OOD rate** — already covered above; a rising trend is an early warning
  independent of both of the above.
- **Prediction-distribution drift** — tracking the distribution of predicted
  classes over time; a sudden shift (e.g. one class disappearing) can signal
  a problem even when input pixel stats look stable.
- **Slow ground truth** — the only way to catch confidently-wrong drift is to
  sample a small percentage of production traffic for human labeling and
  track real accuracy/F1 over time, not just these proxies.

In production, `/monitoring/drift`'s `flags` list is the seed of an alerting
job (scheduled, pushing to Slack/PagerDuty on any flag), with the underlying
`predictions`/`input_stats` tables backing a Grafana-style dashboard.

### 5. Adversarial robustness

No — and this repo demonstrates it rather than just asserting it.
[`scripts/adversarial_demo.py`](scripts/adversarial_demo.py) runs a
single-step, white-box FGSM attack (Goodfellow et al., 2015) at several
perturbation budgets against the trained model on 500 test images, and also
checks whether the OOD autoencoder's flag rate rises on adversarial inputs.
Results (`reports/adversarial_robustness.json` / `.png`) show accuracy
collapsing well before the perturbation becomes visually obvious — expected
for a plain cross-entropy-trained CNN with no adversarial training.

The OOD flag-rate uptick is an interesting free signal but *not* a robust
defense on its own: a white-box attacker aware of the detector can craft
perturbations that fool the classifier while also staying under the
reconstruction-error threshold (a well-documented failure mode of
detect-then-reject defenses). Similarly, the snapshot ensemble helps against
transferred/black-box attacks a little, but doesn't meaningfully help against
a targeted white-box attack, since all snapshots share a training trajectory
and a similar loss landscape.

To actually improve robustness: adversarial training (PGD-based min-max
training, Madry et al.), randomized smoothing for certified guarantees, or at
minimum reporting "robust accuracy at a fixed perturbation budget" against a
multi-step attack (PGD, not just single-step FGSM) as the standard robustness
metric — and evaluating against an attack adaptive to whatever defense is in
place, per Carlini et al.'s guidance on honestly evaluating defenses.

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
- [x] Design discussion (scaling, drift, adversarial robustness)
