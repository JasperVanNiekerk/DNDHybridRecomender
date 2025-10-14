from pathlib import Path
import pandas as pd
from typing import Dict, List
import ast
import numpy as np

def load_mechanical(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)

def build_vocab(series_of_lists: pd.Series) -> Dict[str, int]:
    tokens = series_of_lists.explode().dropna().astype(str)
    uniq = tokens.value_counts().index.tolist()
    return {tok: i for i, tok in enumerate(uniq)}

def _to_set_any(x) -> set:
    # Already a set
    if isinstance(x, set):
        return x
    # Common builtins
    if isinstance(x, (list, tuple)):
        return set(x)
    # Numpy arrays
    if isinstance(x, np.ndarray):
        return set(x.tolist())
    # PyArrow ListScalar / Array
    if hasattr(x, "to_pylist"):
        try:
            return set(x.to_pylist())
        except Exception:
            pass
    # Stringified python list: "['a','b']"
    if isinstance(x, str):
        s = x.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                val = ast.literal_eval(s)
                if isinstance(val, (list, tuple, set)):
                    return set(val)
            except Exception:
                pass
        # Otherwise treat as single token string
        if s:
            return {s}
        return set()
    # NaN / None
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return set()
    # Last resort: try to iterate
    try:
        return set(list(x))
    except Exception:
        return set()

def lists_to_sets(series_of_lists: pd.Series) -> List[set]:
    return [_to_set_any(x) for x in series_of_lists.tolist()]
