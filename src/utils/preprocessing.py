"""
Data Preprocessing for Pharma MMM

Pipeline:
  1. Log-transform all channels and target
  2. Add lagged sample distribution (2-week delayed Rx effect)
  3. Include seasonality index as a direct feature
  4. Temporal train/test split
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

CHANNEL_COLS = [
    "rep_visits",
    "speaker_programs",
    "samples_distributed",
    "emails_sent",
    "ad_clicks",
]

TARGET_COL = "rx_claims"

FEATURE_COLS = [
    "log_rep_visits",
    "log_speaker_programs",
    "log_samples_distributed",
    "log_emails_sent",
    "log_ad_clicks",
    "log_samples_lag2",
    "log_seasonality",
]


def load_and_validate(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["week"])
    required = [TARGET_COL] + CHANNEL_COLS + ["week", "seasonality_index"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df.sort_values("week").reset_index(drop=True)


def log_transform_all(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in CHANNEL_COLS + [TARGET_COL]:
        df[f"log_{col}"] = np.log1p(df[col])
    df["log_seasonality"] = np.log1p(df["seasonality_index"])
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """2-week lag on samples — HCPs write Rx after trialling samples."""
    df = df.copy()
    df["log_samples_lag2"] = df["log_samples_distributed"].shift(2)
    return df.dropna().reset_index(drop=True)


def train_test_split_temporal(df: pd.DataFrame, test_weeks: int = 20) -> tuple:
    """Temporal split — last N weeks held out as test set."""
    return df.iloc[:-test_weeks].copy(), df.iloc[-test_weeks:].copy()


def build_feature_matrix(df: pd.DataFrame) -> tuple:
    """
    Full preprocessing pipeline.
    Returns X (features), y (log Rx claims).
    """
    df = log_transform_all(df)
    df = add_lag_features(df)
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns after preprocessing: {missing}")
    X = df[FEATURE_COLS].astype(float)
    y = df["log_rx_claims"]
    return X, y


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test), scaler
