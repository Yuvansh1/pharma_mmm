"""Tests for Pharma MMM — data, preprocessing, model, scenarios."""

import pytest
import numpy as np
import pandas as pd

from src.utils.data_generator import generate_mmm_data
from src.utils.preprocessing import (
    build_feature_matrix,
    train_test_split_temporal,
    log_transform_all,
    add_lag_features,
    FEATURE_COLS,
)
from src.models.mmm_model import MarketingMixModel


@pytest.fixture(scope="module")
def sample_df():
    return generate_mmm_data(n_weeks=80)


@pytest.fixture(scope="module")
def trained_model(sample_df):
    X, y = build_feature_matrix(sample_df)
    train_df, _ = train_test_split_temporal(pd.concat([X, y], axis=1), test_weeks=12)
    y_col = y.name
    X_train = train_df.drop(columns=[y_col])
    y_train = train_df[y_col]
    model = MarketingMixModel()
    model.fit(X_train, y_train)
    return model, X_train, y_train


class TestDataGenerator:
    def test_shape(self, sample_df):
        assert len(sample_df) == 80

    def test_no_negatives(self, sample_df):
        for col in ["rx_claims", "rep_visits", "samples_distributed", "emails_sent", "ad_clicks"]:
            assert (sample_df[col] >= 0).all()

    def test_required_columns(self, sample_df):
        for col in ["rx_claims", "rep_visits", "speaker_programs",
                    "samples_distributed", "emails_sent", "ad_clicks",
                    "seasonality_index", "event", "week"]:
            assert col in sample_df.columns

    def test_seasonality_range(self, sample_df):
        assert sample_df["seasonality_index"].between(0.9, 1.5).all()


class TestPreprocessing:
    def test_log_transform(self, sample_df):
        df = log_transform_all(sample_df)
        assert "log_rx_claims" in df.columns
        assert (df["log_rx_claims"] >= 0).all()

    def test_lag_feature(self, sample_df):
        df = log_transform_all(sample_df)
        df = add_lag_features(df)
        assert "log_samples_lag2" in df.columns
        assert df["log_samples_lag2"].isnull().sum() == 0

    def test_build_feature_matrix(self, sample_df):
        X, y = build_feature_matrix(sample_df)
        assert len(X) == len(y)
        assert list(X.columns) == FEATURE_COLS
        assert y.name == "log_rx_claims"

    def test_no_nulls_in_features(self, sample_df):
        X, y = build_feature_matrix(sample_df)
        assert X.isnull().sum().sum() == 0

    def test_temporal_split(self, sample_df):
        X, y = build_feature_matrix(sample_df)
        df = pd.concat([X, y], axis=1)
        train, test = train_test_split_temporal(df, test_weeks=12)
        assert len(test) == 12
        assert len(train) == len(df) - 12


class TestMMMModel:
    def test_fits(self, trained_model):
        model, _, _ = trained_model
        assert model.model is not None

    def test_predictions(self, trained_model):
        model, X_train, y_train = trained_model
        preds = model.predict(X_train)
        assert len(preds) == len(X_train)

    def test_train_r2_positive(self, trained_model):
        model, X_train, y_train = trained_model
        metrics = model.evaluate(X_train, y_train)
        assert metrics["r2"] > 0.5, f"Train R2 too low: {metrics['r2']}"

    def test_elasticities(self, trained_model):
        model, _, _ = trained_model
        elast = model.get_elasticities()
        assert isinstance(elast, dict)
        assert len(elast) > 0
        assert "log_rep_visits" in elast

    def test_rep_visits_positive_elasticity(self, trained_model):
        model, _, _ = trained_model
        elast = model.get_elasticities()
        assert elast.get("log_rep_visits", 0) > 0

    def test_contributions(self, trained_model):
        model, X_train, y_train = trained_model
        contrib = model.get_contributions(X_train, y_train)
        assert len(contrib) == len(X_train)
        assert "actual" in contrib.columns
        assert "total_predicted" in contrib.columns

    def test_roi(self, trained_model):
        model, X_train, y_train = trained_model
        contrib = model.get_contributions(X_train, y_train)
        roi = model.compute_roi(contrib, {"log_rep_visits": 100_000})
        assert "log_rep_visits" in roi

    def test_scenario_simulation(self, trained_model):
        model, X_train, _ = trained_model
        result = model.simulate_scenario(X_train, {"log_rep_visits": 1.1})
        assert "base_rx_total" in result
        assert "scenario_rx_total" in result
        assert "delta_pct" in result
        assert result["scenario_rx_total"] != result["base_rx_total"]

    def test_model_summary(self, trained_model):
        model, _, _ = trained_model
        summary = model.summary()
        assert "OLS" in summary
