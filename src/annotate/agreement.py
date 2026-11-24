from typing import Any, Dict, List
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix


def compute_inter_annotator_agreement(
    y1: List[int], y2: List[int], labels: List[int] = [0, 1, 2]
) -> Dict[str, Any]:
    """Computes Cohen's Kappa, raw percentage agreement, and confusion matrix

    between two independent annotators.
    """
    assert len(y1) == len(y2), f"Length mismatch: {len(y1)} vs {len(y2)}"

    n_total = len(y1)
    if n_total == 0:
        return {
            "n_samples": 0,
            "raw_agreement_pct": 0.0,
            "cohen_kappa": 0.0,
            "cohen_kappa_weighted_linear": 0.0,
            "confusion_matrix": {},
        }

    # Raw agreement
    agreements = sum(1 for a, b in zip(y1, y2) if a == b)
    raw_pct = (agreements / n_total) * 100.0

    # Unweighted Cohen's Kappa
    kappa_unweighted = float(cohen_kappa_score(y1, y2, labels=labels))

    # Linear-weighted Cohen's Kappa (penalizes distance of disagreement)
    kappa_weighted = float(cohen_kappa_score(y1, y2, labels=labels, weights="linear"))

    # Confusion Matrix
    cm = confusion_matrix(y1, y2, labels=labels)
    cm_dict = {
        f"true_{labels[i]}": {f"pred_{labels[j]}": int(cm[i, j]) for j in range(len(labels))}
        for i in range(len(labels))
    }

    # Interpretation
    if kappa_unweighted >= 0.81:
        interpretation = "Almost Perfect Agreement"
    elif kappa_unweighted >= 0.61:
        interpretation = "Substantial Agreement"
    elif kappa_unweighted >= 0.41:
        interpretation = "Moderate Agreement"
    elif kappa_unweighted >= 0.21:
        interpretation = "Fair Agreement"
    else:
        interpretation = "Slight / Poor Agreement"

    return {
        "n_samples": n_total,
        "n_agreed": agreements,
        "n_disagreed": n_total - agreements,
        "raw_agreement_pct": round(raw_pct, 2),
        "cohen_kappa": round(kappa_unweighted, 4),
        "cohen_kappa_weighted_linear": round(kappa_weighted, 4),
        "interpretation": interpretation,
        "confusion_matrix": cm_dict,
    }
