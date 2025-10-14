from pathlib import Path
import random, numpy as np
random.seed(42); np.random.seed(42)
import pandas as pd
from collections import Counter, defaultdict

from recs.text import fit_tfidf, nearest_neighbors
from recs.dataio import write_parquet
from recs.class_eligibility import extract_ability_scores, check_requirement

MECH  = Path("processed/mechanical.parquet")
NARR  = Path("processed/narrative.parquet")
CLONG = Path("processed/classes_long.parquet")
OUT   = Path("processed/next_class_hybrid.csv")
OUTX  = Path("processed/next_class_explained.csv")

# weights (tweak if needed)
W_COOCC   = 0.55    # class co-occurrence signal
W_NEIGH   = 0.35    # narrative neighbor votes
W_POP     = 0.10    # popularity prior
PEN_INEL  = 1000.0  # if you want to HARD-BAN ineligible classes, set very large penalty (e.g., 1000)
SOFT_PEN  = 0.35    # OR soft penalty to nudge down ineligible (set PEN_INEL=0 to use this)

def build_class_bags(classes_long: pd.DataFrame):
    bags = defaultdict(set)
    for _, r in classes_long.iterrows():
        bags[int(r["row_id"])].add(str(r["class"]))
    return bags

def class_cooc(bags: dict[int, set[str]]):
    co = defaultdict(Counter)
    for s in bags.values():
        ss = sorted(list(s))
        for i in range(len(ss)):
            for j in range(i+1, len(ss)):
                a, b = ss[i], ss[j]
                co[a][b] += 1
                co[b][a] += 1
    return co

def main():
    mech = pd.read_parquet(MECH)
    narr = pd.read_parquet(NARR)
    cl   = pd.read_parquet(CLONG)

    # vocab & counts
    bags = build_class_bags(cl)
    co   = class_cooc(bags)
    pop_counts = cl["class"].astype(str).value_counts()
    ALL_CLASSES = sorted(pop_counts.index.tolist())

    # narrative tf-idf
    _, X = fit_tfidf(narr)

    rows, details = [], []

    for rid in range(len(mech)):
        owned = set(bags.get(rid, set()))
        primary = str(mech.loc[rid, "primary_class"]) if pd.notna(mech.loc[rid, "primary_class"]) else None

        # 1) co-occ candidates/scores
        co_scores = Counter()
        for c in owned:
            for cand, w in co.get(c, {}).items():
                if cand not in owned:
                    co_scores[cand] += float(w)

        # 2) narrative neighbor class votes
        neigh = nearest_neighbors(X, rid, topn=25)
        neigh_scores = Counter()
        for idx, w in neigh:
            for c in bags.get(idx, set()):
                if c not in owned:
                    neigh_scores[c] += float(w)

        # 3) popularity prior for *all* classes
        maxc = float(pop_counts.max()) if len(pop_counts) else 1.0
        pop_scores = {c: (pop_counts.get(c, 0) / maxc) for c in ALL_CLASSES if c not in owned}

        # blend (sum of weighted signals)
        blended = Counter()
        for k, v in co_scores.items():
            blended[k] += W_COOCC * v
        for k, v in neigh_scores.items():
            blended[k] += W_NEIGH * v
        for k, v in pop_scores.items():
            blended[k] += W_POP * v

        # never suggest current primary (you said “next class to take”)
        if primary in blended:
            del blended[primary]

        # eligibility: extract scores from original snapshot row
        # mech has only mechanical slice; we need the original row for abilities
        # quick read of original snapshot for this row_id
        original_path = Path("processed/original_snapshot.parquet")
        if original_path.exists():
            original = pd.read_parquet(original_path)
            row = original.iloc[rid]
        else:
            # fall back: try mech row only (may miss abilities)
            row = mech.iloc[rid]

        scores = extract_ability_scores(row)

        # apply penalties/bans with explanation
        expl = []
        for cand in list(blended.keys()):
            ok, reason = check_requirement(cand, scores)
            if not ok:
                if PEN_INEL > 0:
                    # hard ban: set enormous negative score
                    blended[cand] = -PEN_INEL
                else:
                    blended[cand] -= SOFT_PEN
            expl.append({
                "row_id": rid,
                "candidate_class": cand,
                "score_pre_sort": float(blended[cand]),
                "eligibility": "eligible" if ok else "ineligible",
                "eligibility_reason": reason,
                "primary_class": primary,
                "owned_classes": "|".join(sorted(list(owned))) if owned else "",
                "from_cooc": float(W_COOCC * co_scores.get(cand, 0.0)),
                "from_narrative": float(W_NEIGH * neigh_scores.get(cand, 0.0)),
                "from_pop": float(W_POP * pop_scores.get(cand, 0.0)),
            })

        # rank & select top-5
        ranked = [c for c, _ in sorted(blended.items(), key=lambda x: x[1], reverse=True)]
        topk = ranked[:5]

        rows.append({
            "row_id": rid,
            "primary_class": primary,
            "owned_classes": "|".join(sorted(list(owned))) if owned else "",
            "top_next_classes": topk
        })

        # keep only explanations for the top-k to reduce size
        keep = {c for c in topk}
        details.extend([d for d in expl if d["candidate_class"] in keep])

    # write outputs
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    pd.DataFrame(details).to_csv(OUTX, index=False)
    print(f"Saved next-class suggestions -> {OUT}")
    print(f"Saved explainability -> {OUTX}")

if __name__ == "__main__":
    main()
