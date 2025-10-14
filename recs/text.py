from typing import Tuple, List, Dict
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def fit_tfidf(narr_df: pd.DataFrame) -> Tuple[TfidfVectorizer, any]:
    vec = TfidfVectorizer(min_df=1, max_df=0.9, ngram_range=(1,2))
    X = vec.fit_transform(narr_df["narrative_text"].fillna(""))
    return vec, X

def nearest_neighbors(X, row_index: int, topn=25) -> List[tuple[int, float]]:
    sims = cosine_similarity(X[row_index], X).ravel()
    order = sims.argsort()[::-1]
    return [(idx, float(sims[idx])) for idx in order if idx != row_index][:topn]

def neighbor_token_scores(
    neighbors: List[tuple[int, float]],
    token_series: pd.Series,
    exclude: set
) -> Dict[str, float]:
    """Turn narrative neighbors into token scores for feats/weapons/armor."""
    scores: Dict[str, float] = {}
    for idx, w in neighbors:
        toks = token_series.iloc[idx]
        if isinstance(toks, list):
            for t in toks:
                if t in exclude: 
                    continue
                scores[t] = scores.get(t, 0.0) + w
    return scores
