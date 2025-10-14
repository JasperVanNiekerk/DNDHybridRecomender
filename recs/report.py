import pandas as pd

def top_k_counts(list_series: pd.Series, k=15) -> pd.Series:
    s = list_series.explode().dropna()
    return s.value_counts().head(k)

def print_basic_report(mech: pd.DataFrame, classes_long: pd.DataFrame) -> None:
    print("=== Rows:", len(mech))
    print("\nTop primary classes:")
    print(mech["primary_class"].value_counts().head(10))

    print("\nTop feats:")
    print(top_k_counts(mech["feats"], k=15))

    print("\nTop weapons:")
    print(top_k_counts(mech["weapons"], k=15))

    print("\nTop armor:")
    print(top_k_counts(mech["armor"], k=15))

    if not classes_long.empty:
        print("\nAvg total level per character:")
        total = classes_long.groupby("row_id")["level"].sum().mean()
        print(round(total, 2))
