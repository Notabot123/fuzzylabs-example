"""End-to-end pipeline: HPO -> final training -> snapshot ensemble -> OOD
autoencoder -> ONNX export -> evaluation report -> drift-monitoring baseline.

Demo-scale by default (see src/config.py). Run with:

    python -m scripts.train_pipeline

Produces (all gitignored except the JSON/PNG reports):
    checkpoints/best_model.pt
    checkpoints/snapshot_{0,1,2}.pt
    checkpoints/ood_autoencoder.pt
    checkpoints/model.onnx
    checkpoints/baseline_input_stats.json
    reports/best_model_report.json, reports/best_model_confusion_matrix.{csv,png}
    reports/ensemble_report.json, reports/ensemble_confusion_matrix.{csv,png}
"""

import json
import sys
import time
from pathlib import Path

import optuna
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CHECKPOINT_DIR, DEVICE, ROOT_DIR, Config, CIFAR10_CLASSES
from src.data import get_datasets, get_dataloaders
from src.ensemble import train_snapshot_ensemble
from src.evaluate import (
    collect_predictions,
    ensemble_predict_proba_fn,
    evaluate_predictions,
    plot_confusion_matrix,
    save_report,
    single_model_predict_proba_fn,
)
from src.hpo import run_hpo
from src.monitoring.drift import compute_baseline_input_stats
from src.onnx_export import export_to_onnx, verify_onnx
from src.ood import calibrate_threshold, save_ood_artifacts, train_autoencoder
from src.train import save_checkpoint, train_model

REPORTS_DIR = ROOT_DIR / "reports"


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cfg = Config()
    log(f"device: {DEVICE}")

    log("loading CIFAR-10 (downloads on first run)...")
    train_ds, val_ds, test_ds = get_datasets(cfg.data)
    _, val_loader, test_loader = get_dataloaders(cfg.data)
    log(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    # 1. HPO
    log(f"running HPO: {cfg.hpo.n_trials} trials x {cfg.hpo.epochs_per_trial} epochs "
        f"on {cfg.hpo.train_subset_size}-image subsets...")
    best_model_cfg, study = run_hpo(cfg, train_ds, val_loader, device=DEVICE)
    log(f"best HPO val_acc={study.best_value:.4f} params={study.best_params}")

    # 2. Final training on the full train split with the best config
    full_train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg.data.batch_size, shuffle=True, num_workers=cfg.data.num_workers
    )
    log(f"final training: {cfg.train.final_epochs} epochs on full train split...")
    model, optimizer, history = train_model(
        best_model_cfg, full_train_loader, val_loader, cfg.train.final_epochs, device=DEVICE
    )
    for h in history:
        log(f"  epoch {h['epoch']}: train_acc={h['train_acc']:.4f} val_acc={h['val_acc']:.4f}")
    save_checkpoint(model, best_model_cfg, CHECKPOINT_DIR / "best_model.pt", extra={"history": history})
    log(f"saved {CHECKPOINT_DIR / 'best_model.pt'}")

    # 3. Snapshot ensemble via warm restarts, continuing from the best model
    log(f"warm-restart snapshot ensemble: {cfg.train.n_snapshots} snapshots "
        f"over {cfg.train.warm_restart_epochs} epochs...")
    snapshot_paths = train_snapshot_ensemble(
        model, optimizer, best_model_cfg, full_train_loader, val_loader, cfg.train, device=DEVICE
    )
    log(f"saved snapshots: {[p.name for p in snapshot_paths]}")

    # 4. Evaluate best single model vs. ensemble on the held-out test set
    log("evaluating best model and ensemble on test set...")
    from src.ensemble import load_ensemble

    ensemble_models = load_ensemble(snapshot_paths, device=DEVICE)

    for name, fn in [
        ("best_model", single_model_predict_proba_fn(model, device=DEVICE)),
        ("ensemble", ensemble_predict_proba_fn(ensemble_models, device=DEVICE)),
    ]:
        y_pred, y_true = collect_predictions(fn, test_loader)
        report, cm = evaluate_predictions(y_true, y_pred)
        save_report(report, cm, REPORTS_DIR, name)
        plot_confusion_matrix(cm, CIFAR10_CLASSES, REPORTS_DIR / f"{name}_confusion_matrix.png")
        log(f"  {name}: accuracy={report['accuracy']:.4f} macro_f1={report['macro avg']['f1-score']:.4f}")

    # 5. OOD autoencoder
    log(f"training OOD autoencoder ({cfg.ood.ae_epochs} epochs)...")
    ae = train_autoencoder(cfg.ood, cfg.data, device=DEVICE)
    threshold = calibrate_threshold(ae, cfg.data, cfg.ood, device=DEVICE)
    save_ood_artifacts(ae, threshold, cfg.ood)
    log(f"OOD threshold (p{cfg.ood.threshold_percentile}) = {threshold:.5f}")

    # 6. ONNX export
    log("exporting best model to ONNX...")
    onnx_path = export_to_onnx(CHECKPOINT_DIR / "best_model.pt")
    ok = verify_onnx(CHECKPOINT_DIR / "best_model.pt", onnx_path)
    log(f"ONNX export {'verified OK' if ok else 'MISMATCH vs PyTorch output'} -> {onnx_path}")

    # 7. Baseline input stats for drift monitoring.
    # Computed on raw [0, 1] pixels (ToTensor, no normalization) to match the
    # contract of api.inference's compute_image_stats(x01) call on live traffic.
    log("computing baseline input stats for drift monitoring...")
    from torchvision import datasets, transforms

    raw_train = datasets.CIFAR10(
        root=str(cfg.data.data_dir), train=True, download=True, transform=transforms.ToTensor()
    )
    baseline_loader = torch.utils.data.DataLoader(raw_train, batch_size=256, shuffle=False)
    baseline_stats = compute_baseline_input_stats(baseline_loader)
    baseline_path = CHECKPOINT_DIR / "baseline_input_stats.json"
    baseline_path.write_text(json.dumps(baseline_stats, indent=2))
    log(f"saved {baseline_path}")

    log("pipeline complete.")


if __name__ == "__main__":
    main()
