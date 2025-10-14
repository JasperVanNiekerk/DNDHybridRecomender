import re
from typing import List, Dict, Optional
from slugify import slugify

CLASS_CHUNK_RE = re.compile(
    r"""
    (?P<class>[A-Za-z][A-Za-z\s'-]*?)          # class
    (?:\s*\(\s*(?P<subclass>[^)]+)\s*\))?      # optional (Subclass)
    \s*Level\s*(?P<level>\d{1,2})              # Level N
    """,
    re.VERBOSE | re.IGNORECASE,
)

def parse_classes_field(classes_str: str) -> List[Dict]:
    """
    Parse class strings like:
    'fighter (Battle Master) Level 5 | wizard (War Magic) Level 3'
    -> [{class, subclass, level}]
    """
    if not isinstance(classes_str, str) or not classes_str.strip():
        return []

    parts = re.split(r"\s*\|\s*", classes_str.strip())
    out = []
    for part in parts:
        m = CLASS_CHUNK_RE.search(part)
        if not m:
            continue
        cls = m.group("class").strip()
        sub = (m.group("subclass") or "").strip()
        lvl = int(m.group("level"))
        out.append({
            "class": slugify(cls, separator="_"),
            "subclass": slugify(sub, separator="_") if sub else None,
            "level": lvl,
        })
    return out

def split_listish(cell: Optional[str]) -> List[str]:
    """
    Split list-like cells that may use commas, pipes, or semicolons.
    Returns slugs.
    """
    if not isinstance(cell, str):
        return []
    tokens = re.split(r"\s*[|,;/]\s*", cell.strip())
    return [slugify(t, separator="_") for t in tokens if t]
