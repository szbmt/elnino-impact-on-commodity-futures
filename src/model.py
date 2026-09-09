import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from xgboost import XGBClassifier, XGBRegressor


class PurgedGroupTimeSeriesSplit:
    """Time series splitter with purging window between train and test folds."""

    def __init__(self, n_splits: int = 5, purge_window: int = 3):
        """Store the number of folds and the size of the purge gap."""
        self.n_splits = n_splits
        self.purge_window = purge_window

    def split(self, X: pd.DataFrame):
        """Yield (train_idx, test_idx) pairs, leaving a purge gap after each train fold to avoid lookahead leakage."""
        n_samples = len(X)
        fold_size = (n_samples - self.purge_window) // (self.n_splits + 1)

        for i in range(self.n_splits):
            train_end = fold_size * (i + 1)
            test_start = train_end + self.purge_window
            test_end = test_start + fold_size

            if test_start >= n_samples:
                break

            train_idx = list(range(train_end))
            test_idx = list(range(test_start, min(test_end, n_samples)))

            if test_idx:
                yield train_idx, test_idx


def evaluate_predictions(
    y_true: pd.Series, y_pred: np.ndarray, is_classification: bool = False
) -> dict:
    """Compute performance metrics for regression or classification."""
    if is_classification:
        acc = accuracy_score(y_true, y_pred) * 100
        return {"Accuracy_%": acc}

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    hit_ratio = np.mean(np.sign(y_true) == np.sign(y_pred)) * 100
    return {"RMSE": rmse, "MAE": mae, "Hit_Ratio_%": hit_ratio}


def calculate_baselines(y_true: pd.Series) -> dict:
    """Calculate naive Buy & Hold and Persistence baseline hit ratios."""
    buy_hold_hit = np.mean(y_true > 0) * 100
    persistence_preds = np.sign(y_true.shift(1)).fillna(1)
    persistence_hit = np.mean(np.sign(y_true) == persistence_preds) * 100

    return {
        "Buy_Hold_Hit_%": buy_hold_hit,
        "Persistence_Hit_%": persistence_hit,
    }


def run_walk_forward_cv(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    task: str = "regression",
    n_splits: int = 5,
    purge_window: int = 3,
) -> tuple[pd.DataFrame, dict]:
    """Run walk-forward cross-validation using XGBoost."""
    splitter = PurgedGroupTimeSeriesSplit(n_splits=n_splits, purge_window=purge_window)
    X, y = df[feature_cols], df[target_col]

    if task == "classification":
        y = (y > 0).astype(int)

    results = []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(X)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        if task == "regression":
            model = XGBRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.03,
                random_state=42,
                n_jobs=-1,
                enable_categorical=True,
            )
        else:
            model = XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.03,
                random_state=42,
                n_jobs=-1,
                enable_categorical=True,
            )

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        fold_df = pd.DataFrame(
            {
                "Date": y_test.index,
                "y_true": y_test.values,
                "y_pred": preds,
                "macro_regime": df.iloc[test_idx].get("macro_regime", "N/A"),
                "fold": fold,
            }
        ).set_index("Date")

        results.append(fold_df)

    predictions_df = pd.concat(results)
    overall_metrics = evaluate_predictions(
        predictions_df["y_true"],
        predictions_df["y_pred"],
        is_classification=(task == "classification"),
    )

    return predictions_df, overall_metrics


def evaluate_by_regime(
    predictions_df: pd.DataFrame, is_classification: bool = False
) -> pd.DataFrame:
    """Evaluate predictions broken down by macro regime."""
    regime_results = {
        regime: evaluate_predictions(
            group["y_true"], group["y_pred"], is_classification
        )
        for regime, group in predictions_df.groupby("macro_regime")
    }
    return pd.DataFrame(regime_results).T
