import numpy as np
from collections import Counter
from typing import List, Dict, Iterable
from math import log

def topn_popularity(train_sets: List[set], n=100) -> List[str]:
    c = Counter()
    for s in train_sets:
        c.update(s)
    return [t for t, _ in c.most_common(n)]

def jaccard_scores(target_items: Iterable[str], cooc: Dict[str, Counter]) -> Dict[str, float]:
    scores = Counter()
    target = set(target_items)
    for t in target:
        if t not in cooc: 
            continue
        for cand, w in cooc[t].items():
            if cand in target: 
                continue
            scores[cand] += w
    return dict(scores)

def build_cooccurrence(train_sets: List[set]) -> Dict[str, Counter]:
    # co-occur counts for item–item (symmetric)
    cooc: Dict[str, Counter] = {}
    for s in train_sets:
        items = list(s)
        for i in range(len(items)):
            a = items[i]
            cooc.setdefault(a, Counter())
            for j in range(i+1, len(items)):
                b = items[j]
                cooc.setdefault(b, Counter())
                cooc[a][b] += 1
                cooc[b][a] += 1
    return cooc

def build_item_stats(train_sets: List[set]):
    item_count = Counter()
    pair_count = Counter()
    n_users = len(train_sets)
    for s in train_sets:
        for a in s:
            item_count[a] += 1
        items = sorted(list(s))
        for i in range(len(items)):
            for j in range(i+1, len(items)):
                pair_count[(items[i], items[j])] += 1
    return n_users, item_count, pair_count

def recommend_itemknn_pmi(known: set, item_count: Counter, pair_count: Counter, k=5):
    scores = Counter()
    N = sum(item_count.values())
    for a in known:
        ca = item_count[a]
        if ca == 0: 
            continue
        for (x, y), cxy in pair_count.items():
            if a != x and a != y:
                continue
            b = y if a == x else x
            if b in known:
                continue
            cb = item_count[b]
            # PMI ~ log p(a,b)/(p(a)p(b)) ; use counts proxy + smoothing
            num = cxy + 1.0
            den = (ca * cb) + 1.0
            pmi = log(num / den)
            scores[b] += pmi
    ranked = [it for it, _ in scores.most_common() if it not in known][:k]
    return ranked

def recommend_popularity(pop_list: List[str], known: set, k=5) -> List[str]:
    out = []
    for it in pop_list:
        if it not in known:
            out.append(it)
        if len(out) >= k:
            break
    return out

def recommend_itemknn(known: set, cooc: Dict[str, Counter], k=5) -> List[str]:
    scores = jaccard_scores(known, cooc)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    out = [it for it, _ in ranked if it not in known][:k]
    return out

