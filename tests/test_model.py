import numpy as np
import pandas as pd
import pytest

from src.model import (
    PurgedGroupTimeSeriesSplit,
    evaluate_by_regime,
    evaluate_predictions,
    run_walk_forward_cv,
)


@pytest.fixture
def sample_model_data():
    """Build a 100-month synthetic DataFrame with two features, a target, and a regime label."""
    dates = pd.date_range(start="2020-01-01", periods=100, freq="MS", name="Date")
    np.random.seed(42)
    return pd.DataFrame(
        {
            "feature_1": np.random.randn(100),
            "feature_2": np.random.randn(100),
            "target": np.random.randn(100),
            "macro_regime": ["Regime_A"] * 50 + ["Regime_B"] * 50,
        },
        index=dates,
    )


def test_purged_group_time_series_split(sample_model_data):
    """Each test fold should start exactly purge_window steps after its train fold ends."""
    splitter = PurgedGroupTimeSeriesSplit(n_splits=3, purge_window=2)
    splits = list(splitter.split(sample_model_data))

    assert len(splits) == 3
    for train_idx, test_idx in splits:
        # Purge window check: test fold must start purge_window steps after train fold ends
        assert test_idx[0] == train_idx[-1] + 1 + 2


def test_evaluate_predictions():
    """Hit ratio should reflect the fraction of correctly predicted return directions."""
    y_true = pd.Series([0.1, -0.2, 0.3])
    y_pred = np.array([0.05, -0.1, -0.1])  # 2 correct directions, 1 wrong

    metrics = evaluate_predictions(y_true, y_pred)
    assert "RMSE" in metrics
    assert "Hit_Ratio_%" in metrics
    assert metrics["Hit_Ratio_%"] == pytest.approx(66.666, rel=1e-2)


def test_run_walk_forward_cv(sample_model_data):
    """Walk-forward CV should return non-empty predictions and regression metrics."""
    features = ["feature_1", "feature_2"]
    preds_df, metrics = run_walk_forward_cv(
        sample_model_data,
        feature_cols=features,
        target_col="target",
        task="regression",
        n_splits=3,
    )

    assert not preds_df.empty
    assert "y_true" in preds_df.columns
    assert "y_pred" in preds_df.columns
    assert "RMSE" in metrics


def test_evaluate_by_regime(sample_model_data):
    """Per-regime metrics should be computed from walk-forward predictions."""
    preds_df, _ = run_walk_forward_cv(
        sample_model_data,
        feature_cols=["feature_1"],
        target_col="target",
        task="regression",
        n_splits=2,
    )

    regime_df = evaluate_by_regime(preds_df)
    assert not regime_df.empty
    assert "RMSE" in regime_df.columns
