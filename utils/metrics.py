"""
utils/metrics.py
-----------------
Evaluation metrics for multi-label anomaly classification.

Computes per-class AUROC, Precision, Recall and F1, plus macro-averages,
which are the metrics that matter most for a triage/screening tool:
- Recall (sensitivity) is prioritized clinically — missing a true anomaly
  (false negative) is far worse than an unnecessary review (false positive).
- AUROC gives a threshold-independent view of separability per finding.
"""

from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)

from utils.logger import get_logger

logger = get_logger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
    threshold: float = 0.5,
) -> Dict:
    """
    Args:
        y_true: (N, C) binary ground-truth array
        y_prob: (N, C) predicted probabilities in [0, 1]
        class_names: list of length C
        threshold: probability cutoff for converting to binary predictions

    Returns:
        dict with per-class and macro-averaged metrics
    """
    y_pred = (y_prob >= threshold).astype(int)

    per_class = {}
    aurocs, precisions, recalls, f1s = [], [], [], []

    for i, name in enumerate(class_names):
        col_true = y_true[:, i]
        col_prob = y_prob[:, i]
        col_pred = y_pred[:, i]

        # AUROC is undefined if a class has only one label value present
        # (common with small validation splits) — guard against that.
        if len(np.unique(col_true)) < 2:
            auroc = float("nan")
            logger.warning(
                "Skipping AUROC for class '%s': only one class present in y_true.",
                name,
            )
        else:
            auroc = roc_auc_score(col_true, col_prob)
            aurocs.append(auroc)

        precision = precision_score(col_true, col_pred, zero_division=0)
        recall = recall_score(col_true, col_pred, zero_division=0)
        f1 = f1_score(col_true, col_pred, zero_division=0)

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

        per_class[name] = {
            "auroc": auroc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    macro = {
        "macro_auroc": float(np.mean(aurocs)) if aurocs else float("nan"),
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
    }

    return {"per_class": per_class, "macro": macro}


def format_metrics_report(metrics: Dict) -> str:
    """Pretty-prints the metrics dict for console / log output."""
    lines = ["\nEvaluation Report", "=" * 60]
    for name, vals in metrics["per_class"].items():
        lines.append(
            f"{name:<18} AUROC={vals['auroc']:.4f}  "
            f"P={vals['precision']:.4f}  R={vals['recall']:.4f}  F1={vals['f1']:.4f}"
        )
    lines.append("-" * 60)
    macro = metrics["macro"]
    lines.append(
        f"{'MACRO AVG':<18} AUROC={macro['macro_auroc']:.4f}  "
        f"P={macro['macro_precision']:.4f}  R={macro['macro_recall']:.4f}  "
        f"F1={macro['macro_f1']:.4f}"
    )
    return "\n".join(lines)
