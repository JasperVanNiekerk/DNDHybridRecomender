from pathlib import Path
import random, numpy as np
random.seed(42); np.random.seed(42)
import pandas as pd

from recs.vocab import load_mechanical, lists_to_sets
from recs.baselines import (
    topn_popularity, build_cooccurrence, build_item_stats,
    recommend_popularity, recommend_itemknn, recommend_itemknn_pmi
)
from recs.evaluate import loo_eval_per_field

MECH = Path("processed/mechanical.parquet")

def run_field(name: str, series: pd.Series):
    # sets + quick debug
    JUNK = {"", "none", "n_a", "na", "n", "weapon", "armor", "unarmed"}
    raw_sets = lists_to_sets(series)
    sets = [{t for t in s if t not in JUNK} for s in raw_sets]
    nonempty = sum(1 for s in sets if len(s) > 0)
    avg_len = (sum(len(s) for s in sets) / max(1, len(sets)))
    print(f"[{name}] rows={len(sets)} nonempty={nonempty} avg_len={avg_len:.2f}")

    # split
    idxs = list(range(len(sets)))
    random.seed(42)
    random.shuffle(idxs)
    split = int(0.8 * len(idxs))
    train_ids, test_ids = idxs[:split], idxs[split:]
    train_sets = [sets[i] for i in train_ids]
    test_sets  = [sets[i] for i in test_ids]

    # models
    pop_list = topn_popularity(train_sets, n=200)
    cooc     = build_cooccurrence(train_sets)
    n_users, item_count, pair_count = build_item_stats(train_sets)

    def rec_pop(known, k=5):
        return recommend_popularity(pop_list, known, k)

    # your version with fallback is perfect — keeping the same behavior
    def rec_knn(known, k=5):
        if not known:
            return recommend_popularity(pop_list, known, k)
        out = recommend_itemknn(known, cooc, k)
        if not out:
            return recommend_popularity(pop_list, known, k)
        return out

    # PMI-based variant with the same fallback behavior
    def rec_pmi(known, k=5):
        if not known:
            return recommend_popularity(pop_list, known, k)
        out = recommend_itemknn_pmi(known, item_count, pair_count, k)
        if not out:
            return recommend_popularity(pop_list, known, k)
        return out

    # evaluate
    r_pop, m_pop, n_pop = loo_eval_per_field(test_sets, rec_pop, k=5)
    r_knn, m_knn, n_knn = loo_eval_per_field(test_sets, rec_knn, k=5)
    r_pmi, m_pmi, n_pmi = loo_eval_per_field(test_sets, rec_pmi, k=5)

    print(f"{name:8} -> Pop  Recall@5: {r_pop:.3f} | MRR@5: {m_pop:.3f}  (n={n_pop})")
    print(f"{'':8}    ItemKNN Recall@5: {r_knn:.3f} | MRR@5: {m_knn:.3f}  (n={n_knn})")
    print(f"{'':8}    PMI    Recall@5: {r_pmi:.3f} | MRR@5: {m_pmi:.3f}  (n={n_pmi})")

def main():
    mech = load_mechanical(MECH)
    print("=== Baseline LOO @5 ===")
    run_field("feats",   mech["feats"])
    run_field("weapons", mech["weapons"])
    run_field("armor",   mech["armor"])

if __name__ == "__main__":
    main()
