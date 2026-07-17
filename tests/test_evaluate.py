import numpy as np

from src.evaluate import evaluate_predictions


def test_evaluate_predictions_perfect_predictions():
    y_true = np.array([0, 1, 2, 3] * 3)
    y_pred = y_true.copy()
    report, cm = evaluate_predictions(y_true, y_pred, class_names=[str(i) for i in range(10)])
    assert report["accuracy"] == 1.0
    assert cm.trace() == cm.sum()  # everything on the diagonal


def test_evaluate_predictions_confusion_matrix_shape_matches_classes():
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 2, 2])
    report, cm = evaluate_predictions(y_true, y_pred, class_names=[str(i) for i in range(10)])
    assert cm.shape == (10, 10)
    assert report["accuracy"] == 2 / 3
