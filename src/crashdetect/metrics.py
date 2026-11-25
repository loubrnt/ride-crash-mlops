"""Metrics shared by training (cross-validation) and evaluation (test set)."""

from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


def classification_metrics(y_true, proba, threshold: float) -> dict[str, float]:
    predicted = proba >= threshold
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "avg_precision": float(average_precision_score(y_true, proba)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
    }
