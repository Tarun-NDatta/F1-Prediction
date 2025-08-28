from typing import List, Hashable, Dict, Tuple
import numpy as np

# Longest Successive Denominator (LSD)

def calculate_lsd_score(
    predicted_sequence: List[Hashable],
    actual_sequence: List[Hashable],
    penalty_factor: float = 2.0,
    max_penalty: float = 30.0,
) -> Dict:
    """Compute LSD between two driver-order sequences (contiguous longest common block).

    Returns keys: lcs, lcs_length, lcs_percentage, position_offset, errors,
    normalized_score, total_drivers, start_predicted, start_actual.
    """
    actual_set = set(actual_sequence)
    pred = [x for x in predicted_sequence if x in actual_set]
    act = list(actual_sequence)

    total = len(act)
    if total == 0 or len(pred) == 0:
        return {
            'lcs': [], 'lcs_length': 0, 'lcs_percentage': 0.0,
            'position_offset': 0, 'errors': 0, 'normalized_score': 0.0,
            'total_drivers': 0, 'start_predicted': None, 'start_actual': None,
        }

    idx_map: Dict[Hashable, int] = {}
    for j, token in enumerate(act):
        if token not in idx_map:
            idx_map[token] = j

    best_len = 0
    best_i = None
    best_j = None

    for i in range(len(pred)):
        token = pred[i]
        if token not in idx_map:
            continue
        j = idx_map[token]
        k = 0
        while i + k < len(pred) and j + k < len(act) and pred[i + k] == act[j + k]:
            k += 1
        if k > best_len or (k == best_len and best_i is not None and abs(i - j) < abs(best_i - best_j)):
            best_len = k
            best_i = i
            best_j = j

    if best_len == 0:
        lcs = []
        lcs_pct = 0.0
        pos_offset = 0
        errors = total
        score = 0.0
        start_pred = None
        start_act = None
    else:
        lcs = pred[best_i: best_i + best_len]
        lcs_pct = (best_len / total) * 100.0
        pos_offset = abs(best_i - best_j)
        errors = max(total - best_len, 0)
        offset_pen = min(pos_offset * penalty_factor, max_penalty)
        score = max(0.0, lcs_pct - offset_pen)
        start_pred, start_act = int(best_i), int(best_j)

    return {
        'lcs': lcs,
        'lcs_length': int(best_len),
        'lcs_percentage': round(lcs_pct, 1) if best_len else 0.0,
        'position_offset': int(pos_offset),
        'errors': int(errors),
        'normalized_score': round(score, 1),
        'total_drivers': int(total),
        'start_predicted': start_pred,
        'start_actual': start_act,
    }


def kendall_tau(pred: List[Hashable], actual: List[Hashable]) -> float:
    """Kendall's tau between predicted and actual orders over common set."""
    try:
        from scipy.stats import kendalltau
        rank_actual = {d: i for i, d in enumerate(actual)}
        common = [d for d in pred if d in rank_actual]
        if len(common) < 2:
            return float('nan')
        x = list(range(len(common)))
        y = [rank_actual[d] for d in common]
        return float(kendalltau(x, y, nan_policy='omit').correlation)
    except Exception:
        rank_actual = {d: i for i, d in enumerate(actual)}
        y = [rank_actual[d] for d in pred if d in rank_actual]
        n = len(y)
        if n < 2:
            return float('nan')
        conc = disc = 0
        for i in range(n):
            for j in range(i+1, n):
                if y[i] < y[j]: conc += 1
                elif y[i] > y[j]: disc += 1
        total_pairs = n*(n-1)//2
        return (conc - disc) / total_pairs if total_pairs else float('nan')


def spearman_rho(pred: List[Hashable], actual: List[Hashable]) -> float:
    """Spearman's rho between predicted and actual ranks over common set."""
    try:
        from scipy.stats import spearmanr
        rank_actual = {d: i for i, d in enumerate(actual)}
        common = [d for d in pred if d in rank_actual]
        if len(common) < 2:
            return float('nan')
        x = list(range(len(common)))
        y = [rank_actual[d] for d in common]
        return float(spearmanr(x, y, nan_policy='omit').correlation)
    except Exception:
        rank_actual = {d: i for i, d in enumerate(actual)}
        common = [d for d in pred if d in rank_actual]
        if len(common) < 2:
            return float('nan')
        x = np.arange(len(common), dtype=float)
        y = np.array([rank_actual[d] for d in common], dtype=float)
        x = (x - x.mean()) / (x.std() or 1.0)
        y = (y - y.mean()) / (y.std() or 1.0)
        return float(np.mean(x * y))

