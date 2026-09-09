import numpy as np
import pandas as pd
import pytest

from src.features import (
    add_forward_targets,
    add_macro_features,
    add_oni_lags,
    add_rolling_volatility,
    assign_macro_regimes,
    build_feature_pipeline,
    compute_log_returns,
)


@pytest.fixture
def sample_df():
    """Build a 30-month synthetic DataFrame with prices, ONI, and macro columns."""
    dates = pd.date_range(start="2020-01-01", periods=30, freq="MS", name="Date")
    return pd.DataFrame(
        {
            "cocoa_price": [2000 + i * 50 for i in range(30)],
            "gold_price": [1500 + i * 10 for i in range(30)],
            "ONI": [0.5 + (i % 3) * 0.1 for i in range(30)],
            "us_interest_rate": [1.5 + i * 0.1 for i in range(30)],
            "usd_index": [95.0 + i * 0.2 for i in range(30)],
        },
        index=dates,
    )


def test_compute_log_returns(sample_df):
    """Log returns should be NaN on the first row and correct on later rows."""
    df_res = compute_log_returns(sample_df, ["cocoa_price"])
    assert "cocoa_price_log_return" in df_res.columns
    assert np.isnan(df_res["cocoa_price_log_return"].iloc[0])
    assert df_res["cocoa_price_log_return"].iloc[1] == pytest.approx(
        np.log(2050 / 2000)
    )


def test_add_oni_lags(sample_df):
    """Lagged ONI columns should shift values back by the requested number of months."""
    df_res = add_oni_lags(sample_df, lags=(3,))
    assert "oni_lag_3m" in df_res.columns
    assert np.isnan(df_res["oni_lag_3m"].iloc[2])
    assert df_res["oni_lag_3m"].iloc[3] == sample_df["ONI"].iloc[0]


def test_add_macro_features(sample_df):
    """Rate/USD level, diff, and lag columns should all be created."""
    df_res = add_macro_features(sample_df, lags=(1,))
    assert "us_rate_diff_1m" in df_res.columns
    assert "us_rate_lag_1m" in df_res.columns
    assert "usd_index_log_return" in df_res.columns
    assert "usd_index_lag_1m" in df_res.columns


def test_assign_macro_regimes(sample_df):
    """Dates should be bucketed into named regimes with one-hot columns."""
    df_res = assign_macro_regimes(sample_df)
    assert "macro_regime" in df_res.columns
    assert "3. COVID Shock" in df_res.columns
    assert df_res["3. COVID Shock"].sum() > 0


def test_add_rolling_volatility(sample_df):
    """A rolling std column should be added for the given return column."""
    df_res = compute_log_returns(sample_df, ["cocoa_price"])
    df_res = add_rolling_volatility(df_res, ["cocoa_price_log_return"], windows=(3,))
    assert "cocoa_price_log_return_vol_3m" in df_res.columns


def test_add_forward_targets(sample_df):
    """The forward target should be NaN at the end of the series (no future data)."""
    df_res = add_forward_targets(sample_df, ["cocoa_price"], horizons=(1,))
    assert "target_cocoa_price_1m" in df_res.columns
    assert np.isnan(df_res["target_cocoa_price_1m"].iloc[-1])


def test_build_feature_pipeline(sample_df):
    """The full pipeline should run end-to-end and drop all remaining NaNs."""
    df_features = build_feature_pipeline(
        sample_df, price_cols=["cocoa_price", "gold_price"]
    )
    assert not df_features.empty
    assert "cocoa_price_log_return" in df_features.columns
    assert "oni_lag_3m" in df_features.columns
    assert "us_rate_lag_1m" in df_features.columns
    assert "target_cocoa_price_1m" in df_features.columns
    assert df_features.isna().sum().sum() == 0
