from typing import List, Dict, Set

HEAVY_ARMOR_TOKENS = {
    "plate_armor", "half_plate", "splint", "ring_mail", "chain_mail",
    "dwarven_plate", "dragon_scale_mail_black", "dragon_scale_mail_gold"
}
HEAVY_OK = {"fighter", "paladin"}  # simple rule-of-thumb

def legality_penalties(primary_class: str | None, candidates: List[str]) -> Dict[str, float]:
    """Return additive penalties (negative) per candidate; 0 means no penalty."""
    penalties: Dict[str, float] = {}
    pcls = (primary_class or "").lower()
    for c in candidates:
        pen = 0.0
        if c in HEAVY_ARMOR_TOKENS and pcls not in HEAVY_OK:
            pen -= 0.25  # soft nudge down, not a hard ban
        penalties[c] = pen
    return penalties

def apply_penalties(scores: Dict[str, float], penalties: Dict[str, float]) -> Dict[str, float]:
    out = dict(scores)
    for k, pen in penalties.items():
        if k in out:
            out[k] += pen
    return out

def remove_duplicates(scores: Dict[str, float], owned: Set[str]) -> Dict[str, float]:
    return {k: v for k, v in scores.items() if k not in owned}
