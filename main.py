"""
Main pipeline execution script for commodity price prediction.
Evaluates XGBoost models against Buy & Hold and Persistence baselines.
"""

import os

import pandas as pd

from src.data_loader import load_merged_dataset
from src.features import build_feature_pipeline
from src.model import (
    calculate_baselines,
    evaluate_by_regime,
    run_walk_forward_cv,
)
from src.visualization import plot_predictions_vs_actual, plot_regime_performance


def main():
    """Load data, build features, run walk-forward CV per commodity, and save the summary + plots."""
    df = load_merged_dataset()
    price_cols = [
        "cocoa_price",
        "coffee_price",
        "gold_price",
        "soybean_price",
        "wheat_price",
        "natural_gas_price",
    ]
    df_features = build_feature_pipeline(df, price_cols=price_cols)

    os.makedirs("results/plots", exist_ok=True)
    summary_results = []

    for commodity in price_cols:
        target_col = f"target_{commodity}_1m"
        if target_col not in df_features.columns:
            continue

        regime_columns = [
            "1. Pre-GFC & GFC",
            "2. Low Rate Era",
            "3. COVID Shock",
            "4. Post-COVID Inflation",
        ]

        ignore_cols = (
            [c for c in df_features.columns if c.startswith("target_")]
            + ["macro_regime"]
            + regime_columns
        )
        feature_cols = [c for c in df_features.columns if c not in ignore_cols]

        preds_reg, metrics_reg = run_walk_forward_cv(
            df_features, feature_cols, target_col, task="regression"
        )

        preds_cls, metrics_cls = run_walk_forward_cv(
            df_features, feature_cols, target_col, task="classification"
        )
        regime_cls = evaluate_by_regime(preds_cls, is_classification=True)

        baselines = calculate_baselines(preds_reg["y_true"])

        summary_results.append(
            {
                "Commodity": commodity,
                "XGB_Reg_Hit_%": metrics_reg["Hit_Ratio_%"],
                "XGB_Cls_Acc_%": metrics_cls["Accuracy_%"],
                "Buy_Hold_%": baselines["Buy_Hold_Hit_%"],
                "Persistence_%": baselines["Persistence_Hit_%"],
            }
        )

        plot_predictions_vs_actual(
            preds_reg,
            title=f"{commodity.capitalize()} Return Prediction (XGBoost)",
            save_path=f"results/plots/{commodity}_regression.png",
        )
        plot_regime_performance(
            regime_cls,
            title=f"{commodity.capitalize()} Directional Accuracy by Regime",
            save_path=f"results/plots/{commodity}_classification_regimes.png",
        )

    summary_df = pd.DataFrame(summary_results)
    summary_df = summary_df.round(2)
    summary_df.to_csv("results/summary_metrics.csv", index=False)

    print("--- EVALUATION SUMMARY ---")
    print(summary_df)


if __name__ == "__main__":
    main()
