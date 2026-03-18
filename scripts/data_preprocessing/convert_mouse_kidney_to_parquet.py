"""Convert mouse-kidney raw CSV to Parquet for lower memory footprint.

Usage:
    python scripts/data_preprocessing/convert_mouse_kidney_to_parquet.py

Reads data/mouse-kidney/raw/GSE190094.csv and writes
data/mouse-kidney/raw/GSE190094.parquet alongside it.
"""

from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/mouse-kidney/raw")
CSV_PATH = RAW_DIR / "GSE190094.csv"
PARQUET_PATH = RAW_DIR / "GSE190094.parquet"

if __name__ == "__main__":
    print(f"Reading {CSV_PATH} ...")
    df = pd.read_csv(CSV_PATH)
    print(f"  shape: {df.shape}")

    # Downcast float64 columns to float32 to reduce memory at load time
    float64_cols = df.select_dtypes("float64").columns
    df[float64_cols] = df[float64_cols].astype("float32")
    print(f"  downcast {len(float64_cols)} float64 columns to float32")

    print(f"Writing {PARQUET_PATH} ...")
    df.to_parquet(PARQUET_PATH, index=False)

    csv_size = CSV_PATH.stat().st_size / (1024**3)
    parquet_size = PARQUET_PATH.stat().st_size / (1024**3)
    print(f"  CSV:     {csv_size:.2f} GB")
    print(f"  Parquet: {parquet_size:.2f} GB")
    print(f"  Ratio:   {parquet_size / csv_size:.1%}")
