"""Inspect local CSV files before choosing a training target."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

def main():
    csv_files = sorted(RAW_DIR.rglob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in: {RAW_DIR}")
        print("Place a permitted dataset CSV in data/raw and run this script again.")
        return
    for path in csv_files:
        print("\n" + "=" * 88)
        print(f"FILE: {path.relative_to(ROOT)}")
        try:
            df = pd.read_csv(path, nrows=1000, low_memory=False)
            print(f"Preview shape (up to 1000 rows): {df.shape}")
            print("\nColumns:")
            for col in df.columns:
                print(f"  - {col} (dtype={df[col].dtype}, missing={df[col].isna().sum()})")
            print("\nFirst 5 rows:")
            print(df.head().to_string(max_cols=12, index=False))
        except Exception as exc:
            print(f"Could not read this CSV: {exc}")

if __name__ == "__main__":
    main()
