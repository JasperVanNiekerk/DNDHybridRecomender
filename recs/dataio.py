from pathlib import Path
import pandas as pd

def read_characters_xlsx(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Excel not found: {path}")
    # Read first sheet by default
    df = pd.read_excel(path)
    # Normalize column names: lowercase, snake_case
    df.columns = (
        df.columns
          .str.strip()
          .str.replace(r"[^\w]+", "_", regex=True)
          .str.lower()
    )
    return df

def write_parquet(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
