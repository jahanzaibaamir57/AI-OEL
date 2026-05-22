import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


NUMERIC_COLUMNS = [
    "Year",
    "average_rain_fall_mm_per_year",
    "pesticides_tonnes",
    "avg_temp",
    "hg/ha_yield",
]

CATEGORICAL_COLUMNS = ["Area", "Item"]


class DataPreprocessor:
    def __init__(self) -> None:
        self.numeric_imputer = SimpleImputer(strategy="median")
        self.iqr_bounds: Dict[str, Tuple[float, float]] = {}

    def load_data(self, csv_path: Path) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        unnamed_cols = [c for c in df.columns if c.startswith("Unnamed") or c == ""]
        if "H1" in df.columns:
            unnamed_cols.append("H1")
        if unnamed_cols:
            df = df.drop(columns=list(set(unnamed_cols)))
        return df

    def _fit_outlier_bounds(self, df: pd.DataFrame) -> None:
        for col in NUMERIC_COLUMNS:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            self.iqr_bounds[col] = (float(lower), float(upper))

    def _clip_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        for col, (lower, upper) in self.iqr_bounds.items():
            df[col] = df[col].clip(lower=lower, upper=upper)
        return df

    def fit_transform(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        df = raw_df.copy()

        numeric_df = df[NUMERIC_COLUMNS].copy()
        numeric_df[:] = self.numeric_imputer.fit_transform(numeric_df)
        df[NUMERIC_COLUMNS] = numeric_df

        self._fit_outlier_bounds(df)
        df = self._clip_outliers(df)

        for col in CATEGORICAL_COLUMNS:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace("", "Unknown")

        return df

    def transform(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        df = raw_df.copy()

        numeric_df = df[NUMERIC_COLUMNS].copy()
        numeric_df[:] = self.numeric_imputer.transform(numeric_df)
        df[NUMERIC_COLUMNS] = numeric_df

        df = self._clip_outliers(df)

        for col in CATEGORICAL_COLUMNS:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace("", "Unknown")

        return df

    def save_metadata(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "numeric_columns": NUMERIC_COLUMNS,
            "categorical_columns": CATEGORICAL_COLUMNS,
            "outlier_iqr_bounds": self.iqr_bounds,
        }
        (output_dir / "preprocessing_metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )


def build_data_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        rows.append(
            {
                "column": col,
                "dtype": str(df[col].dtype),
                "missing_values": int(df[col].isna().sum()),
                "unique_values": int(df[col].nunique()),
                "example": str(df[col].iloc[0]) if not df.empty else "",
            }
        )
    return pd.DataFrame(rows)
