import os

import matplotlib.pyplot as plt
import pandas as pd


def plot_predictions_vs_actual(
    preds_df: pd.DataFrame, title: str, save_path: str | None = None
) -> None:
    """Plot actual vs predicted returns over time."""
    plt.figure(figsize=(12, 5))
    plt.plot(preds_df.index, preds_df["y_true"], label="Actual", alpha=0.7)
    plt.plot(
        preds_df.index,
        preds_df["y_pred"],
        label="Predicted",
        linestyle="--",
        color="red",
    )
    plt.axhline(0, color="black", linewidth=0.8, linestyle=":")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Value / Return")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_regime_performance(
    regime_metrics_df: pd.DataFrame, title: str, save_path: str | None = None
) -> None:
    """Plot directional accuracy across macro regimes compared to random baseline."""
    plt.figure(figsize=(8, 4))
    metric_col = (
        "Accuracy_%" if "Accuracy_%" in regime_metrics_df.columns else "Hit_Ratio_%"
    )

    plt.bar(
        regime_metrics_df.index,
        regime_metrics_df[metric_col],
        color="teal",
        alpha=0.8,
    )
    plt.axhline(50, color="red", linestyle="--", label="Random Guess (50%)")
    plt.title(title)
    plt.ylabel(metric_col)
    plt.xlabel("Macro Regime")
    plt.ylim(0, 100)
    plt.xticks(rotation=15)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
