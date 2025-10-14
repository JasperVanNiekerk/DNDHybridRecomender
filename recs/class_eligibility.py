from typing import Dict, Tuple

# PHB multiclass ability minima (Artificer included for completeness)
REQS = {
    "barbarian": {"str": 13},
    "bard": {"cha": 13},
    "cleric": {"wis": 13},
    "druid": {"wis": 13},
    "fighter": {"str_or_dex": 13},          # STR or DEX 13
    "monk": {"dex": 13, "wis": 13},
    "paladin": {"str": 13, "cha": 13},
    "ranger": {"dex": 13, "wis": 13},
    "rogue": {"dex": 13},
    "sorcerer": {"cha": 13},
    "warlock": {"cha": 13},
    "wizard": {"int": 13},
    "artificer": {"int": 13},
}

# Map possible column names to short ability keys
ABILITY_ALIASES = {
    "strength": "str", "str": "str",
    "dexterity": "dex", "dex": "dex",
    "constitution": "con", "con": "con",
    "intelligence": "int", "int": "int",
    "wisdom": "wis", "wis": "wis",
    "charisma": "cha", "cha": "cha",
}

def _coerce_int(x):
    try:
        return int(x)
    except Exception:
        return None

def extract_ability_scores(row) -> Dict[str, int]:
    """
    Try common shapes:
    - flattened columns like abilityscores_strength, ability_scores_strength, etc.
    - or a JSON-like dict column 'abilityscores'/'ability_scores'
    Returns dict with keys: str,dex,con,int,wis,cha
    Missing become None.
    """
    out = {k: None for k in ["str","dex","con","int","wis","cha"]}
    # 1) flattened columns
    for col in row.index:
        lc = str(col).lower()
        for key, short in ABILITY_ALIASES.items():
            if lc.endswith(f"_{key}") or lc == key or ("ability" in lc and key in lc):
                val = _coerce_int(row[col])
                if val is not None:
                    out[short] = val
    # 2) embedded dict
    for candidate in ["abilityscores", "ability_scores"]:
        if candidate in row.index:
            val = row[candidate]
            if isinstance(val, dict):
                for k, v in val.items():
                    s = ABILITY_ALIASES.get(str(k).lower())
                    if s:
                        iv = _coerce_int(v)
                        if iv is not None:
                            out[s] = iv
    return out

def check_requirement(klass: str, scores: Dict[str,int]) -> Tuple[bool, str]:
    """
    Returns (is_eligible, reason)
    """
    k = klass.lower()
    rules = REQS.get(k, {})
    if not rules:
        return True, "no_requirement"
    # handle fighter special 'str_or_dex'
    if "str_or_dex" in rules:
        need = rules["str_or_dex"]
        ok = ((scores.get("str") or 0) >= need) or ((scores.get("dex") or 0) >= need)
        return (ok, f"need STR>=13 or DEX>=13; have STR={scores.get('str')}, DEX={scores.get('dex')}")
    # all others: AND of present minima
    missing = []
    for abbr, need in rules.items():
        have = scores.get(abbr) or 0
        if have < need:
            missing.append(f"{abbr.upper()}>={need} (have {have})")
    if missing:
        return False, " & ".join(missing)
    return True, "ok"
