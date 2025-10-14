from pathlib import Path
import pandas as pd
from collections import Counter, defaultdict

from recs.text import fit_tfidf, nearest_neighbors
from recs.features import normalize  # for parsing if needed
from recs.dataio import write_parquet

CLONG = Path("processed/classes_long.parquet")
NARR  = Path("processed/narrative.parquet")
MECH  = Path("processed/mechanical.parquet")
OUT   = Path("processed/next_class.csv")

def build_class_bags(classes_long: pd.DataFrame):
    bags = defaultdict(set)
    for _, r in classes_long.iterrows():
        bags[int(r["row_id"])].add(str(r["class"]))
    return bags

def class_cooc(bags: dict[int, set[str]]):
    co = defaultdict(Counter)
    for s in bags.values():
        ss = list(s)
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

    bags = build_class_bags(cl)
    co   = class_cooc(bags)
    _, X = fit_tfidf(narr)

    ALL_CLASSES = sorted(set(cl["class"].astype(str)))

    rows = []
    for rid in range(len(mech)):
        owned = bags.get(rid, set())
        primary = str(mech.loc[rid, "primary_class"]) if pd.notna(mech.loc[rid, "primary_class"]) else None

        # 1) Co-oc scores based on owned classes (if single-class, this is empty)
        co_scores = Counter()
        for c in owned:
            for cand, w in co[c].items():
                if cand not in owned:
                    co_scores[cand] += w

        # 2) Narrative neighbor class votes
        neigh = nearest_neighbors(X, rid, topn=25)
        neigh_scores = Counter()
        for idx, w in neigh:
            for c in bags.get(idx, set()):
                if c not in owned:
                    neigh_scores[c] += w

        # 3) Popularity prior
        pop = Counter()
        for c, cnt in cl["class"].value_counts().items():
            if c not in owned:
                pop[c] = cnt

        # Blend (simple weights)
        scores = Counter()
        for d, w in [(co_scores, 0.6), (neigh_scores, 0.3), (pop, 0.1)]:
            for k, v in d.items():
                scores[k] += w * float(v)

        # Remove the current primary class from suggestions
        if primary in scores:
            del scores[primary]

        ranked = [c for c, _ in scores.most_common()][:5]
        rows.append({
            "row_id": rid,
            "primary_class": primary,
            "owned_classes": sorted(list(owned)),
            "top_next_classes": ranked
        })

    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"Saved next-class suggestions -> {OUT}")

if __name__ == "__main__":
    main()
