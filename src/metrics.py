"""Image-level anomaly-detection metrics.

Metrics are reported per defect type against the shared ``good`` test images as
the negative set, plus an ``overall`` group (good vs every anomaly). Generic
over any dataset whose samples carry a ``defect_type`` string: MVTec AD's many
named defect types, or a two-way split, are both just values of that string.
"""

from typing import Dict, List, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def max_f1(labels: np.ndarray, scores: np.ndarray) -> float:
    """Best F1 over all thresholds (AnomalyDINO-style)."""
    precision, recall, _ = precision_recall_curve(labels, scores)
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = (2 * precision * recall) / (precision + recall)
    return float(np.max(f1[np.isfinite(f1)]))


def _binary_scores(
    labels: np.ndarray, scores: np.ndarray
) -> Dict[str, float]:
    # Undefined (not zero) when a group has only one class; NaN lets the run
    # finish instead of sklearn raising.
    lab = np.asarray(labels)
    if lab.size == 0 or lab.min() == lab.max():
        return {"auroc": float("nan"), "ap": float("nan"), "f1": float("nan")}
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "ap": float(average_precision_score(labels, scores)),
        "f1": max_f1(labels, scores),
    }


def compute_image_metrics(
    scores: Sequence[float],
    labels: Sequence[int],
    defect_types: Sequence[str],
    good_label: str = "good",
) -> Dict[str, Dict[str, float]]:
    """Image-level AUROC, AP and max-F1, overall and per defect type.

    ``scores`` are per-image anomaly scores (higher is more anomalous),
    ``labels`` are 0/1 (0 = good) and ``defect_types`` are the test folder
    names. Returns ``{group: {"auroc", "ap", "f1", "n_pos", "n_neg"}}``.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    defect_types = np.asarray(defect_types)

    good = defect_types == good_label
    n_good = int(good.sum())
    results: Dict[str, Dict[str, float]] = {}

    # Overall: good vs everything anomalous.
    results["overall"] = {**_binary_scores(labels, scores),
                          "n_pos": int((labels == 1).sum()), "n_neg": n_good}

    # Per anomaly type, against the shared good negatives.
    for atype in sorted(set(defect_types) - {good_label}):
        sel = good | (defect_types == atype)
        y = (defect_types[sel] == atype).astype(np.int64)
        results[atype] = {**_binary_scores(y, scores[sel]),
                          "n_pos": int((defect_types == atype).sum()),
                          "n_neg": n_good}

    return results


def format_metrics(results: Dict[str, Dict[str, float]]) -> str:
    """Compact multi-line table for logging."""
    keys = ["overall"] + sorted(k for k in results if k != "overall")
    lines = [f"{'group':<22} {'AUROC':>7} {'AP':>7} {'F1':>7}  (pos/neg)"]
    for k in keys:
        m = results[k]
        auroc = f"{m.get('auroc', float('nan')):.4f}"
        ap = f"{m['ap']:.4f}" if "ap" in m else "   -   "
        f1 = f"{m['f1']:.4f}" if "f1" in m else "   -   "
        counts = f"({m['n_pos']}/{m['n_neg']})" if "n_pos" in m else ""
        lines.append(f"{k:<22} {auroc:>7} {ap:>7} {f1:>7}  {counts}")
    return "\n".join(lines)


def average_over_categories(
    per_category: Dict[str, Dict[str, Dict[str, float]]]
) -> Dict[str, Dict[str, float]]:
    """Average each metric across categories, group by group.

    ``per_category`` maps category -> the dict returned by
    :func:`compute_image_metrics`. Returns the same nested shape with means.
    """
    groups: Dict[str, Dict[str, List[float]]] = {}
    for cat_results in per_category.values():
        for group, metrics in cat_results.items():
            for name, value in metrics.items():
                if name.startswith("n_"):
                    continue
                groups.setdefault(group, {}).setdefault(name, []).append(value)
    return {
        group: {name: float(np.mean(vals)) for name, vals in metrics.items()}
        for group, metrics in groups.items()
    }
