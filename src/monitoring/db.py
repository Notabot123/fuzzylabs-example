"""SQLite-backed logging of inference confidences and input-distribution stats.

Two tables:
  - predictions: one row per inference, the predicted class + softmax
    confidence (+ full distribution) and whether it was flagged OOD. Rolling
    aggregates over this table reveal *model* drift (confidence collapsing,
    a class disappearing/dominating, OOD rate climbing).
  - input_stats: one row per inference, summary statistics of the raw input
    tensor (mean, std, skewness, kurtosis per channel, averaged). Rolling
    aggregates over this table reveal *data* drift (input distribution
    shifting away from what the model was trained/calibrated on) even before
    it shows up in model confidence.

Both are intentionally denormalized/simple (one row per request) rather than
pre-aggregated, so drift queries can pick any time window after the fact.
"""

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

from src.config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    model_name TEXT NOT NULL,
    predicted_class TEXT NOT NULL,
    confidence REAL NOT NULL,
    probs_json TEXT NOT NULL,
    is_ood INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS input_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    mean REAL NOT NULL,
    std REAL NOT NULL,
    skewness REAL NOT NULL,
    kurtosis REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp);
CREATE INDEX IF NOT EXISTS idx_input_stats_timestamp ON input_stats(timestamp);
"""


@contextmanager
def get_connection(db_path: Path = DB_PATH):
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        yield conn
    finally:
        conn.close()


def init_db(db_path: Path = DB_PATH):
    with get_connection(db_path) as conn:
        conn.executescript(SCHEMA)
        conn.commit()


def log_prediction(
    model_name: str,
    predicted_class: str,
    confidence: float,
    probs: list,
    is_ood: bool,
    db_path: Path = DB_PATH,
):
    with get_connection(db_path) as conn:
        conn.execute(
            "INSERT INTO predictions (timestamp, model_name, predicted_class, confidence, probs_json, is_ood) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (time.time(), model_name, predicted_class, confidence, json.dumps(probs), int(is_ood)),
        )
        conn.commit()


def log_input_stats(mean: float, std: float, skewness: float, kurtosis: float, db_path: Path = DB_PATH):
    with get_connection(db_path) as conn:
        conn.execute(
            "INSERT INTO input_stats (timestamp, mean, std, skewness, kurtosis) VALUES (?, ?, ?, ?, ?)",
            (time.time(), mean, std, skewness, kurtosis),
        )
        conn.commit()
