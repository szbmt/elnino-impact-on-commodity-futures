import numpy as np
import pandas as pd


def compute_log_returns(df: pd.DataFrame, price_cols: list[str]) -> pd.DataFrame:
    """Calculates monthly log returns for specified price columns."""
    df_out = df.copy()
    for col in price_cols:
        if col in df_out.columns:
            df_out[f"{col}_log_return"] = np.log(df_out[col] / df_out[col].shift(1))
    return df_out


def add_oni_lags(
    df: pd.DataFrame, oni_col: str = "ONI", lags: tuple[int, ...] = (3, 6, 9, 12)
) -> pd.DataFrame:
    """Generates time-lagged features for the ONI climate signal."""
    df_out = df.copy()
    if oni_col in df_out.columns:
        for lag in lags:
            df_out[f"oni_lag_{lag}m"] = df_out[oni_col].shift(lag)
    return df_out


def add_macro_features(
    df: pd.DataFrame, lags: tuple[int, ...] = (1, 3, 6)
) -> pd.DataFrame:
    """Adds level and lagged features for US Interest Rate and USD Index."""
    df_out = df.copy()

    if "us_interest_rate" in df_out.columns:
        df_out["us_rate_diff_1m"] = df_out["us_interest_rate"].diff(1)
        for lag in lags:
            df_out[f"us_rate_lag_{lag}m"] = df_out["us_interest_rate"].shift(lag)

    if "usd_index" in df_out.columns:
        df_out["usd_index_log_return"] = np.log(
            df_out["usd_index"] / df_out["usd_index"].shift(1)
        )
        for lag in lags:
            df_out[f"usd_index_lag_{lag}m"] = df_out["usd_index"].shift(lag)

    return df_out


def assign_macro_regimes(df: pd.DataFrame) -> pd.DataFrame:
    """Categorizes dates into macro regimes and applies one-hot encoding."""
    df_out = df.copy()
    conditions = [
        (df_out.index >= "2000-09-01") & (df_out.index <= "2012-12-31"),
        (df_out.index >= "2013-01-01") & (df_out.index <= "2019-12-31"),
        (df_out.index >= "2020-01-01") & (df_out.index <= "2022-12-31"),
        (df_out.index >= "2023-01-01"),
    ]
    regime_names = [
        "1. Pre-GFC & GFC",
        "2. Low Rate Era",
        "3. COVID Shock",
        "4. Post-COVID Inflation",
    ]

    df_out["macro_regime"] = np.select(conditions, regime_names, default="Other")
    regime_dummies = pd.get_dummies(df_out["macro_regime"], dtype=int)
    df_out = pd.concat([df_out, regime_dummies], axis=1)
    return df_out


def add_rolling_volatility(
    df: pd.DataFrame, return_cols: list[str], windows: tuple[int, ...] = (3, 6)
) -> pd.DataFrame:
    """Calculates rolling standard deviations for log returns."""
    df_out = df.copy()
    for col in return_cols:
        if col in df_out.columns:
            for window in windows:
                df_out[f"{col}_vol_{window}m"] = (
                    df_out[col].rolling(window=window).std()
                )
    return df_out


def add_forward_targets(
    df: pd.DataFrame, price_cols: list[str], horizons: tuple[int, ...] = (1, 3)
) -> pd.DataFrame:
    """Generates forward-looking target variables for model training."""
    df_out = df.copy()
    for col in price_cols:
        if col in df_out.columns:
            for horizon in horizons:
                df_out[f"target_{col}_{horizon}m"] = np.log(
                    df_out[col].shift(-horizon) / df_out[col]
                )
    return df_out


def build_feature_pipeline(df: pd.DataFrame, price_cols: list[str]) -> pd.DataFrame:
    """Executes the complete feature engineering pipeline."""
    df_features = compute_log_returns(df, price_cols)
    return_cols = [
        f"{col}_log_return"
        for col in price_cols
        if f"{col}_log_return" in df_features.columns
    ]

    df_features = add_oni_lags(df_features)
    df_features = add_macro_features(df_features)
    df_features = assign_macro_regimes(df_features)
    df_features = add_rolling_volatility(df_features, return_cols)
    df_features = add_forward_targets(df_features, price_cols)

    return df_features.dropna()
