"""Compute per-image input statistics and summarize drift from logged history.

`compute_image_stats` is called on every inference request (cheap: a handful
of scalar reductions over the input tensor). `drift_summary` is called on
demand (e.g. by a monitoring endpoint or a scheduled job) to compare a recent
window of logged requests against a baseline computed once from the training
distribution.
"""

import sqlite3
from dataclasses import dataclass

import numpy as np
import torch
from scipy.stats import kurtosis, skew

import src.monitoring.db as db  # not `from ... import DB_PATH`: keeps a single
# source of truth for the DB location (db.DB_PATH), so patching it in tests
# (or a future config reload) is visible to both modules.


def compute_image_stats(x: torch.Tensor) -> dict:
    """x: a single image tensor (3, H, W) or a batch (B, 3, H, W), any numeric range.

    Stats are computed over all pixel values (flattened across channels/space,
    and batch if present) — cheap distributional fingerprint of the input(s).
    """
    values = x.detach().cpu().numpy().reshape(-1).astype(np.float64)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "skewness": float(skew(values)),
        "kurtosis": float(kurtosis(values)),
    }


@dataclass
class DriftReport:
    n_samples: int
    recent_confidence_mean: float
    recent_ood_rate: float
    recent_input_stats: dict
    baseline_input_stats: dict
    input_stat_z_scores: dict
    flags: list


def _fetch_recent(conn: sqlite3.Connection, table: str, columns: list, limit: int):
    cols = ", ".join(columns)
    rows = conn.execute(
        f"SELECT {cols} FROM {table} ORDER BY timestamp DESC LIMIT ?", (limit,)
    ).fetchall()
    return rows


def drift_summary(
    baseline_input_stats: dict,
    db_path=None,
    window: int = 200,
    z_threshold: float = 3.0,
) -> DriftReport:
    """Compare the most recent `window` logged requests against a baseline.

    `baseline_input_stats` should be the mean/std of each stat (mean, std,
    skewness, kurtosis) computed once over the training set — see
    `compute_baseline_input_stats` below, invoked from
    `scripts/train_pipeline.py`. A z-score beyond `z_threshold` on any stat,
    or a rising OOD rate / falling confidence, is surfaced as a flag for a
    human (or an alert) to investigate.
    """
    db_path = db_path if db_path is not None else db.DB_PATH
    conn = sqlite3.connect(str(db_path))
    try:
        pred_rows = _fetch_recent(conn, "predictions", ["confidence", "is_ood"], window)
        stat_rows = _fetch_recent(conn, "input_stats", ["mean", "std", "skewness", "kurtosis"], window)
    finally:
        conn.close()

    flags = []
    if not pred_rows or not stat_rows:
        return DriftReport(0, 0.0, 0.0, {}, baseline_input_stats, {}, ["no logged requests yet"])

    confidences = np.array([r[0] for r in pred_rows])
    ood_flags = np.array([r[1] for r in pred_rows])
    recent_confidence_mean = float(confidences.mean())
    recent_ood_rate = float(ood_flags.mean())

    stat_names = ["mean", "std", "skewness", "kurtosis"]
    recent_stats = {
        name: float(np.mean([r[i] for r in stat_rows])) for i, name in enumerate(stat_names)
    }

    z_scores = {}
    for name in stat_names:
        baseline_mean = baseline_input_stats[name]["mean"]
        baseline_std = max(baseline_input_stats[name]["std"], 1e-8)
        z = (recent_stats[name] - baseline_mean) / baseline_std
        z_scores[name] = float(z)
        if abs(z) > z_threshold:
            flags.append(f"input '{name}' drifted {z:+.1f} std from baseline")

    if recent_ood_rate > 0.2:
        flags.append(f"OOD rate elevated: {recent_ood_rate:.0%} of last {len(pred_rows)} requests")
    if recent_confidence_mean < 0.5:
        flags.append(f"mean softmax confidence low: {recent_confidence_mean:.2f}")

    return DriftReport(
        n_samples=len(pred_rows),
        recent_confidence_mean=recent_confidence_mean,
        recent_ood_rate=recent_ood_rate,
        recent_input_stats=recent_stats,
        baseline_input_stats=baseline_input_stats,
        input_stat_z_scores=z_scores,
        flags=flags,
    )


def compute_baseline_input_stats(loader) -> dict:
    """Per-image stats over a whole dataset, returning {stat_name: {mean, std}}.

    Run once (offline, against the training set) to calibrate what "normal"
    input statistics look like; the result is what `drift_summary` compares
    live traffic against.
    """
    per_image = {"mean": [], "std": [], "skewness": [], "kurtosis": []}
    for x, _ in loader:
        for i in range(x.shape[0]):
            stats = compute_image_stats(x[i])
            for k, v in stats.items():
                per_image[k].append(v)
    return {
        name: {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        for name, vals in per_image.items()
    }
