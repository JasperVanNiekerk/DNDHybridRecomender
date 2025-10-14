from pathlib import Path
from recs.dataio import read_characters_xlsx, write_parquet
from recs.features import normalize
from recs.report import print_basic_report

RAW_XLSX = Path("data/raw/characters.xlsx")
OUT_DIR  = Path("processed")

def main():
    df = read_characters_xlsx(RAW_XLSX)
    tables = normalize(df)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_parquet(tables["mechanical"], OUT_DIR / "mechanical.parquet")
    write_parquet(tables["classes_long"], OUT_DIR / "classes_long.parquet")
    write_parquet(tables["narrative"], OUT_DIR / "narrative.parquet")
    write_parquet(tables["original"], OUT_DIR / "original_snapshot.parquet")

    print("Saved to /processed")
    print_basic_report(tables["mechanical"], tables["classes_long"])

if __name__ == "__main__":
    main()
