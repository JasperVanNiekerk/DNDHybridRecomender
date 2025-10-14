from typing import List, Dict, Tuple
from collections import defaultdict
import numpy as np
import random

def recall_at_k(target: str, recs: List[str], k=5) -> float:
    return 1.0 if target in recs[:k] else 0.0

def mrr_at_k(target: str, recs: List[str], k=5) -> float:
    for i, r in enumerate(recs[:k]):
        if r == target:
            return 1.0 / (i + 1)
    return 0.0

def loo_eval_per_field(
    all_sets: List[set],
    recommender_fn,
    k=5
) -> Tuple[float, float, int]:
    """
    Leave-One-Out:
    For each example with size>=2, hide one item, recommend with remaining.
    recommender_fn(known_set, k) -> List[str]
    """
    r_list, m_list = [], []
    n = 0
    for s in all_sets:
        if len(s) < 1:
            continue
        target = random.choice(list(s))  # arbitrary held-out; rotate for randomness if you want
        known = set(s) - {target}
        recs = recommender_fn(known, k)
        r_list.append(recall_at_k(target, recs, k))
        m_list.append(mrr_at_k(target, recs, k))
        n += 1
    recall = float(np.mean(r_list)) if r_list else 0.0
    mrr    = float(np.mean(m_list)) if m_list else 0.0
    return recall, mrr, n

def loo_eval_rowwise(
    all_sets: List[set],
    recommender_for_row,  # fn(row_id:int, known:set, k:int) -> List[str]
    k=5
) -> Tuple[float, float, int]:
    r_list, m_list = [], []
    n = 0
    for rid, s in enumerate(all_sets):
        if len(s) < 1:
            continue
        target = random.choice(list(s))
        known = set(s) - {target}
        recs = recommender_for_row(rid, known, k)
        r_list.append(recall_at_k(target, recs, k))
        m_list.append(mrr_at_k(target, recs, k))
        n += 1
    recall = float(np.mean(r_list)) if r_list else 0.0
    mrr    = float(np.mean(m_list)) if m_list else 0.0
    return recall, mrr, n
