from typing import List, Dict, Tuple
from collections import Counter
import math

def blend_scores(*score_dicts: Dict[str, float], weights: List[float] | None = None, topn: int = 5) -> List[str]:
    if weights is None:
        weights = [1.0] * len(score_dicts)
    acc = Counter()
    for w, sd in zip(weights, score_dicts):
        for k, v in sd.items():
            acc[k] += w * v
    ranked = [k for k, _ in acc.most_common() if not math.isnan(acc[k])]
    return ranked[:topn]

def blend_with_attribution(parts, weights):
    """
    parts: list[dict[item->score]] in same order as weights
    returns: final_scores, contribs (dict[item->{part_i:score}])
    """
    final = {}
    contribs = {}
    for i, (sd, w) in enumerate(zip(parts, weights)):
        for k, v in sd.items():
            final[k] = final.get(k, 0.0) + w*float(v)
            contribs.setdefault(k, {})[f"part_{i}"] = w*float(v)
    return final, contribs
