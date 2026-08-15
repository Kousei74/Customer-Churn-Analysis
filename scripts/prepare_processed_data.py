import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data.preprocess import preprocess_data
from src.features.build_features import build_features

RAW = "Telco-Customer-Churn-ML/data/raw/Telco-Customer-Churn.csv"
OUT = "Telco-Customer-Churn-ML/data/processed/telco_churn_processed.csv"


def main():
    raw_path = next((p for p in RAW if p.exists()), None)
    if raw_path is None:
        raise FileNotFoundError("No raw Telco CSV found in data/raw/")

    df = pd.read_csv(raw_path)
    df = preprocess_data(df, target_col="Churn")

    if "Churn" in df.columns and df["Churn"].dtype == "object":
        df["Churn"] = df["Churn"].str.strip().map({"No": 0, "Yes": 1}).astype("Int64")

    assert df["Churn"].isna().sum() == 0, "Churn has NaNs after preprocess"
    assert set(df["Churn"].unique()) <= {0, 1}, "Churn not 0/1 after preprocess"

    df_processed = build_features(df, target_col="Churn")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df_processed.to_csv(OUT, index=False)
    print(f"✅ Processed dataset saved to {OUT} | Shape: {df_processed.shape}")


if __name__ == "__main__":
    main()
