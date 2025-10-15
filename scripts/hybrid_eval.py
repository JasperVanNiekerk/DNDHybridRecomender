from pathlib import Path
import random, numpy as np
random.seed(42); np.random.seed(42)
import pandas as pd
from collections import Counter

from recs.vocab import load_mechanical, lists_to_sets
from recs.baselines import topn_popularity, build_cooccurrence, recommend_popularity, recommend_itemknn
from recs.evaluate import loo_eval_per_field, loo_eval_rowwise
from recs.text import fit_tfidf, nearest_neighbors, neighbor_token_scores
from recs.hybrid import blend_with_attribution
from recs.legal import legality_penalties, apply_penalties, remove_duplicates
from recs.tune import sample_simplex, save_weights, load_weights
WEIGHTS_FILE = Path("processed/hybrid_item_weights.json")
TUNE_TRIALS  = 120

MECH = Path("processed/mechanical.parquet")
NARR = Path("processed/narrative.parquet")
OUT  = Path("processed/recommendations.csv")

# weights for blending (tweakable)
W_ITEMKNN = 0.5
W_NEIGH   = 0.4
W_POP     = 0.1

def weights_for(field):
    # default
    wi, wn, wp = 0.5, 0.4, 0.1
    if field == "feats":
        # feats benefit more from narrative; item-item can be sparse
        wi, wn, wp = 0.35, 0.55, 0.10
    return wi, wn, wp



def make_sets(series: pd.Series):
    raw = lists_to_sets(series)
    JUNK = {"", "none", "n_a", "na", "n", "weapon", "armor", "unarmed"}
    return [{t for t in s if t not in JUNK} for s in raw]

def eval_field(name: str, mech: pd.DataFrame, X, narr_df: pd.DataFrame):
    series = mech[name]
    sets   = make_sets(series)
    global_counts = Counter()
    for s in sets:
        global_counts.update(s)
    global_vocab = set(global_counts.keys())
    # train/test split for baselines
    idxs = list(range(len(sets)))
    random.seed(42)
    random.shuffle(idxs)
    split = int(0.8 * len(idxs))
    train_ids, test_ids = idxs[:split], idxs[split:]
    train_sets = [sets[i] for i in train_ids]
    test_sets  = [sets[i] for i in test_ids]

    # popularity + cooc
    pop_list = topn_popularity(train_sets, n=300)
    cooc     = build_cooccurrence(train_sets)

    def tune_weights_for_field():
        best = (-1.0, (0.5, 0.4, 0.1))  # (recall, weights)
        # sample candidates (mix deterministic grid + random dirichlet)
        cand = sample_simplex(n=3, num=TUNE_TRIALS, kind="dirichlet") + sample_simplex(n=3, num=0, kind="grid")
        for w in cand:
            rec_hyb = make_rec_hybrid_for_row(w)
            def wrapper(rid: int, known: set, k=5):
                items, _ = rec_hyb(rid, k=k, _known_override=known)
                return items
            r_hyb, m_hyb, n_hyb = loo_eval_rowwise(sets, wrapper, k=5)
            if r_hyb > best[0]:
                best = (r_hyb, w)
        return best  # (best_recall, (w_i, w_n, w_p))


    def rec_pop(known, k=5): 
        return recommend_popularity(pop_list, known, k)

    def rec_itemknn(known, k=5):
        if not known:
            return recommend_popularity(pop_list, known, k)
        out = recommend_itemknn(known, cooc, k)
        return out or recommend_popularity(pop_list, known, k)

    # Hybrid recommender: itemknn + narrative neighbors + popularity (+ legality)
    token_series = series  # feats/weapons/armor lists per row

    def make_rec_hybrid_for_row(weights_tuple):
        # unpack & freeze the weights
        w_i, w_n, w_p = map(float, weights_tuple)

        def _rec(row_id: int, k=5, _known_override=None, _w_i=w_i, _w_n=w_n, _w_p=w_p):
            known = sets[row_id] if _known_override is None else _known_override
            known = {str(x) for x in known}

            # 1) itemknn pool
            itemknn_scores = Counter()
            if known:
                for it in rec_itemknn(known, k=80):
                    itemknn_scores[str(it)] += 1.0

            # 2) narrative neighbors
            neigh = nearest_neighbors(X, row_id, topn=35)
            neigh_scores = neighbor_token_scores(neigh, token_series, exclude=known)
            neigh_scores = {str(t): float(w) for t, w in neigh_scores.items()}

            # 3) global prior over ALL tokens
            maxc = max(global_counts.values()) if global_counts else 1
            pop_scores = {str(it): global_counts[it] / maxc for it in global_vocab}

            # blend with attribution (use captured weights)
            parts = [itemknn_scores, neigh_scores, pop_scores]
            blended, contribs = blend_with_attribution(parts, [_w_i, _w_n, _w_p])

            # legality
            primary = str(mech.loc[row_id, "primary_class"]) if pd.notna(mech.loc[row_id, "primary_class"]) else None
            blended = remove_duplicates(blended, owned=known)
            pen_map = {} if name == "feats" else legality_penalties(primary, list(blended.keys()))
            blended = apply_penalties(blended, pen_map)

            ranked = sorted(blended.items(), key=lambda x: x[1], reverse=True)
            topk = ranked[:k]

            details = []
            for item, score in topk:
                c = contribs.get(item, {})
                details.append({
                    "item": item,
                    "score": float(score),
                    "from_itemknn": float(c.get("part_0", 0.0)),
                    "from_narrative": float(c.get("part_1", 0.0)),
                    "from_pop": float(c.get("part_2", 0.0)),
                    "penalty": float(pen_map.get(item, 0.0)),
                    "primary_class": primary
                })
            return [it for it, _ in topk], details

        return _rec


    saved = load_weights(WEIGHTS_FILE, {})
    field_key = f"item::{name}"
    if field_key in saved:
        best_w = tuple(saved[field_key])
    else:
        best_score, best_w = tune_weights_for_field()
        saved[field_key] = list(best_w)
        save_weights(WEIGHTS_FILE, saved)

    rec_hybrid_for_row = make_rec_hybrid_for_row(best_w)

    # Evaluate baselines + hybrid (row-aware)
    print(f"[{name}] rows={len(sets)} nonempty={sum(1 for s in sets if s)} avg_len={sum(len(s) for s in sets)/max(1,len(sets)):.2f}")
    r_pop, m_pop, n_pop   = loo_eval_per_field(test_sets, rec_pop, k=5)
    r_knn, m_knn, n_knn   = loo_eval_per_field(test_sets, rec_itemknn, k=5)
    print(f"{name:8} -> Pop     R@5:{r_pop:.3f} MRR@5:{m_pop:.3f} (n={n_pop})")
    print(f"{'':8}    ItemKNN R@5:{r_knn:.3f} MRR@5:{m_knn:.3f} (n={n_knn})")

    # Row-aware hybrid eval: when we hide the target, we must pass that reduced known set
    def rec_hybrid_rowaware(rid: int, known: set, k=5):
        items, _ = rec_hybrid_for_row(rid, k=k, _known_override=known)
        return items
    r_hyb, m_hyb, n_hyb = loo_eval_rowwise(sets, rec_hybrid_rowaware, k=5)
    print(f"{'':8}    Hybrid* R@5:{r_hyb:.3f} MRR@5:{m_hyb:.3f} (n={n_hyb})  w={tuple(round(x,2) for x in best_w)}")

    return rec_hybrid_for_row

def export_character_recs(mech: pd.DataFrame, rec_fns: dict[str, callable], k=5):
    rows, expl = [], []
    for rid in range(len(mech)):
        row = {
            "row_id": rid,
            "primary_class": mech.loc[rid, "primary_class"],
            "primary_subclass": mech.loc[rid, "primary_subclass"],
        }
        for field, fn in rec_fns.items():
            ret = fn(rid, k=k)
            # Accept list, (items,), (items, details), or longer tuples
            if isinstance(ret, tuple):
                items = ret[0]
                details = ret[1] if len(ret) > 1 else []
            else:
                items = ret
                details = []
            # Normalize items to list
            if not isinstance(items, list):
                items = list(items) if items is not None else []
            row[f"top_{field}"] = items
            # Collect details if present
            if isinstance(details, list):
                for d in details:
                    if isinstance(d, dict):
                        d = dict(d)  # shallow copy
                        d.update({"row_id": rid, "field": field})
                        expl.append(d)
        rows.append(row)
    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    # save explanations (if any)
    OUT_EXPL = OUT.parent / "recommendations_explained.csv"
    if expl:
        pd.DataFrame(expl).to_csv(OUT_EXPL, index=False)
        print(f"\nSaved per-character recommendations -> {OUT}")
        print(f"Saved per-candidate explanations   -> {OUT_EXPL}")
    else:
        print(f"\nSaved per-character recommendations -> {OUT}")


def main():
    mech = load_mechanical(MECH)
    narr = pd.read_parquet(NARR)
    _, X  = fit_tfidf(narr)

    rec_feat   = eval_field("feats",   mech, X, narr)
    rec_weapon = eval_field("weapons", mech, X, narr)
    rec_armor  = eval_field("armor",   mech, X, narr)

    export_character_recs(mech, {
        "feats": rec_feat,
        "weapons": rec_weapon,
        "armor": rec_armor,
    }, k=5)

if __name__ == "__main__":
    main()
