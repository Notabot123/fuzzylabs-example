import torch

import src.monitoring.db as db
import src.monitoring.drift as drift


def test_init_db_creates_expected_tables(tmp_path):
    db_path = tmp_path / "test.db"
    db.init_db(db_path)

    import sqlite3

    conn = sqlite3.connect(str(db_path))
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()
    assert {"predictions", "input_stats"} <= tables


def test_log_prediction_and_input_stats_roundtrip(tmp_path):
    db_path = tmp_path / "test.db"
    db.init_db(db_path)
    db.log_prediction("best_model", "cat", 0.87, [0.87, 0.13], is_ood=False, db_path=db_path)
    db.log_input_stats(mean=0.5, std=0.2, skewness=0.0, kurtosis=-1.0, db_path=db_path)

    import sqlite3

    conn = sqlite3.connect(str(db_path))
    pred_rows = conn.execute("SELECT predicted_class, confidence, is_ood FROM predictions").fetchall()
    stat_rows = conn.execute("SELECT mean, std FROM input_stats").fetchall()
    conn.close()

    assert pred_rows == [("cat", 0.87, 0)]
    assert stat_rows == [(0.5, 0.2)]


def test_compute_image_stats_returns_expected_keys():
    x = torch.rand(3, 32, 32)
    stats = drift.compute_image_stats(x)
    assert set(stats.keys()) == {"mean", "std", "skewness", "kurtosis"}
    assert 0.0 <= stats["mean"] <= 1.0


def test_drift_summary_flags_no_data_when_db_empty(tmp_path):
    db_path = tmp_path / "empty.db"
    db.init_db(db_path)
    report = drift.drift_summary(baseline_input_stats={}, db_path=db_path)
    assert report.n_samples == 0
    assert "no logged requests yet" in report.flags


def test_drift_summary_flags_elevated_ood_rate(tmp_path):
    db_path = tmp_path / "test.db"
    db.init_db(db_path)
    for i in range(10):
        db.log_prediction("m", "cat", 0.9, [0.9, 0.1], is_ood=(i < 5), db_path=db_path)
        db.log_input_stats(mean=0.5, std=0.2, skewness=0.0, kurtosis=-1.0, db_path=db_path)

    baseline = {
        "mean": {"mean": 0.5, "std": 0.05},
        "std": {"mean": 0.2, "std": 0.05},
        "skewness": {"mean": 0.0, "std": 0.05},
        "kurtosis": {"mean": -1.0, "std": 0.05},
    }
    report = drift.drift_summary(baseline, db_path=db_path, window=10)
    assert report.recent_ood_rate == 0.5
    assert any("OOD rate elevated" in f for f in report.flags)
