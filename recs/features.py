import pandas as pd
from typing import Dict, List
from .parsing import parse_classes_field, split_listish

NARRATIVE_FIELDS = ["appearance","backstory","ideals","bonds","flaws","personality"]

def normalize(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    # Expect these columns (but tolerate variants)
    # We'll try to infer reasonable defaults if missing.
    colmap = {c: c for c in df.columns}

    # Classes
    class_col = next((c for c in df.columns if c in ["class","classes","class_es","class_subclass_levels"]), None)
    if class_col is None:
        # fail fast; this one is critical for later evaluation
        raise KeyError("Could not find a 'classes' column. Expected one of: class, classes, class_es, class_subclass_levels")

    # Optional list-ish fields
    feats_col   = next((c for c in df.columns if c in ["feats","feat_list"]), None)
    weapons_col = next((c for c in df.columns if c in ["weapons","weapon_list"]), None)
    armor_col   = next((c for c in df.columns if c in ["armor","armour","armor_list","armour_list"]), None)

    # Parse classes into exploded rows
    parsed = df[[class_col]].copy()
    parsed["__parsed"] = parsed[class_col].apply(parse_classes_field)

    class_rows: List[Dict] = []
    for idx, row in parsed.iterrows():
        for item in row["__parsed"]:
            class_rows.append({
                "row_id": idx,
                "class": item["class"],
                "subclass": item["subclass"],
                "level": item["level"],
            })
    classes_long = pd.DataFrame(class_rows)

    # Primary class = highest level (ties: first)
    if not classes_long.empty:
        prim = (classes_long.sort_values(["row_id","level"], ascending=[True,False])
                          .drop_duplicates("row_id")
                          .rename(columns={"class":"primary_class","subclass":"primary_subclass","level":"primary_level"}))
        prim = prim[["row_id","primary_class","primary_subclass","primary_level"]]
    else:
        prim = pd.DataFrame(columns=["row_id","primary_class","primary_subclass","primary_level"])

    # Build a “mechanical” table (multi-hot-ish lists preserved)
    mech = df.copy()
    mech["row_id"] = mech.index

    if feats_col:   mech["feats"]   = mech[feats_col].apply(split_listish)
    else:           mech["feats"]   = [[] for _ in range(len(mech))]
    if weapons_col: mech["weapons"] = mech[weapons_col].apply(split_listish)
    else:           mech["weapons"] = [[] for _ in range(len(mech))]
    if armor_col:   mech["armor"]   = mech[armor_col].apply(split_listish)
    else:           mech["armor"]   = [[] for _ in range(len(mech))]

    mech = mech.merge(prim, on="row_id", how="left")

    # Narrative table: one concatenated text field + originals
    narrative = df.copy()
    narrative["row_id"] = narrative.index
    for f in NARRATIVE_FIELDS:
        if f not in narrative.columns:
            narrative[f] = ""
    narrative["narrative_text"] = narrative[NARRATIVE_FIELDS].fillna("").agg(" \n".join, axis=1)

    # Slim down “mech_export”
    keep_cols = ["row_id","primary_class","primary_subclass","primary_level","feats","weapons","armor"]
    mech_export = mech[keep_cols].copy()

    return {
        "classes_long": classes_long,   # (row_id, class, subclass, level)
        "mechanical": mech_export,      # (row_id, primary_*, feats[], weapons[], armor[])
        "narrative": narrative[["row_id","narrative_text"] + NARRATIVE_FIELDS],
        "original": df.assign(row_id=df.index),
    }
