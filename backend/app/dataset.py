"""Loads the Pima Indians Diabetes dataset and provides the notebook's train/test split."""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "diabetes.csv"
TARGET = "Outcome"
FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
            "BMI", "DiabetesPedigreeFunction", "Age"]
# A value of 0 is physiologically impossible in these columns: it stands for "not measured"
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
SEED = 42

DESCRIPTIONS = {
    "Pregnancies": "Number of times pregnant",
    "Glucose": "Plasma glucose 2 hours into an oral glucose tolerance test",
    "BloodPressure": "Diastolic blood pressure (mm Hg)",
    "SkinThickness": "Triceps skin fold thickness (mm)",
    "Insulin": "2-hour serum insulin (mu U/ml)",
    "BMI": "Body mass index (kg/m²)",
    "DiabetesPedigreeFunction": "Diabetes likelihood from family history",
    "Age": "Age in years",
    "Outcome": "Target: 1 = diabetic, 0 = not diabetic",
}


@lru_cache(maxsize=1)
def _load() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def get_df() -> pd.DataFrame:
    return _load().copy()


def xy(df: pd.DataFrame, features: list[str] | None = None):
    features = features or FEATURES
    return df[features], df[TARGET]


def zeros_to_nan(df: pd.DataFrame) -> pd.DataFrame:
    """Replace the impossible zeros with NaN so they can be counted and imputed."""
    out = df.copy()
    out[ZERO_AS_MISSING] = out[ZERO_AS_MISSING].replace(0, np.nan)
    return out


def notebook_split(df: pd.DataFrame, test_size: float = 0.2, stratify: bool = True, seed: int = SEED,
                   features: list[str] | None = None):
    """The split from notebook section 5: 80/20, stratified on Outcome, random_state=42."""
    X, y = xy(df, features)
    return train_test_split(X, y, test_size=test_size, stratify=y if stratify else None, random_state=seed)


def column_choices(df: pd.DataFrame) -> dict:
    return {"features": FEATURES, "all": FEATURES + [TARGET], "zero": ZERO_AS_MISSING}
