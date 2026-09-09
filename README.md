# El Niño vs Commodities

This project evaluates whether combining the **Oceanic Niño Index (ONI)** with macroeconomic indicators (US interest rates and USD strength) improves direction prediction for monthly commodity returns. The analysis covers six commodities: cocoa, coffee, soybeans, wheat, natural gas, and gold, comparing machine learning forecasts against simple baseline strategies.

## Main Objective

To test if an XGBoost model using climate and macro features can achieve a higher hit ratio on monthly commodity returns than standard Buy & Hold and Persistence baselines.

## Methodology

**Data**
- Climate signal: NOAA's Oceanic Niño Index (ONI), fetched directly from `cpc.ncep.noaa.gov`.
- Commodity prices and macro series: monthly closes via `yfinance` (cocoa, coffee, soybean, wheat, natural gas, gold futures; 3-month T-bill rate; EUR/USD as a USD proxy).
- Sample period: September 2000 – present, monthly frequency.

**Feature engineering**
- Log returns for every commodity.
- ONI values lagged at 3, 6, 9, and 12 months to account for delayed climate impacts on commodities.
- Macro features: interest rate level, 1-month diff, lags; USD index log returns and lags.
- Past 3- and 6-month rolling volatility of log returns.
- Forward-looking targets (1-month and 3-month forward log returns) as prediction labels.

**Modeling**
- **XGBoost**, trained two ways per commodity:
  - *Regression:* Predicts the exact forward return value. The direction (positive/negative sign) serves as the trading direction.
  - *Classification:* Directly predicts whether the next period return will be positive or negative.
- **Purged walk-forward cross-validation**: A time-series split where the model only trains on past data to predict future test windows. A purge gap is inserted between training and testing to prevent future data from leaking into the training set.

**Baselines**
- **Buy & Hold**: Always assumes a positive return and that the market always goes up.
- **Persistence**: Predicts that next month's return direction will match the current month's direction (captures momentum).

## Repository structure

```
.
├── src/
│   ├── data_loader.py
│   ├── features.py
│   ├── model.py
│   └── visualization.py
├── tests/
│   ├── test_data_loader.py
│   ├── test_features.py
│   └── test_model.py
├── EDA.ipynb
├── main.py
├── requirements.txt
└── results/
    ├── summary_metrics.csv
    └── plots/
```

## Setup & usage

```bash
python -m venv venv
venv\Scripts\activate           # Mac / Linux: source venv/bin/activate
pip install -r requirements.txt

python main.py                  # runs main.py
pytest                          # runs tests
```

## Results

| Commodity        | XGB Reg. Hit % | XGB Cls. Acc % | Buy & Hold % | Persistence % |
|------------------|:--------------:|:--------------:|:------------:|:--------------:|
| Soybean          | 56.67          | 51.25          | 52.08        | 51.25          |
| Cocoa            | 55.00          | 55.00          | 54.17        | 41.67          |
| Coffee           | 52.50          | 48.75          | 49.17        | 45.42          |
| Wheat            | 49.58          | 50.83          | 51.67        | 47.08          |
| Natural Gas      | 50.42          | 49.58          | 49.58        | 53.33          |
| Gold             | 48.33          | 46.67          | 55.42        | 51.67          |

*(Hit ratio = % of months where the predicted return direction matched the actual direction.)*

### Predictions vs. Actuals

<p align="center">
  <img src="/results/plots/soybean_price_regression.png" width="49%">
  <img src="/results/plots/cocoa_price_regression.png" width="49%">
</p>
<p align="center">
  <img src="/results/plots/coffee_price_regression.png" width="49%">
  <img src="/results/plots/wheat_price_regression.png" width="49%">
</p>
<p align="center">
  <img src="/results/plots/natural_gas_price_regression.png" width="49%">
  <img src="/results/plots/gold_price_regression.png" width="49%">
</p>

### Directional Accuracy by Macro Regime

<p align="center">
  <img src="/results/plots/soybean_price_classification_regimes.png" width="49%">
  <img src="/results/plots/cocoa_price_classification_regimes.png" width="49%">
</p>
<p align="center">
  <img src="/results/plots/coffee_price_classification_regimes.png" width="49%">
  <img src="/results/plots/wheat_price_classification_regimes.png" width="49%">
</p>
<p align="center">
  <img src="/results/plots/natural_gas_price_classification_regimes.png" width="49%">
  <img src="/results/plots/gold_price_classification_regimes.png" width="49%">
</p>

## Key Findings

**Where the model outperforms the baselines:**
- **Soybean**: XGBoost hits 56.67%, beating both Buy & Hold (52.08%) and Persistence (51.25%). This is the one where the features track something the baselines miss.
- **Cocoa**: Model accuracy (54.58%) is close to Buy & Hold (54.17%) and beats Persistence (41.67%).
- **Coffee**: Performance is balanced; the regression hit ratio (52.50%) slightly edges out its baselines.

**Where a baseline wins:**
- **Gold**: Buy & Hold wins at 55.42% vs. the model's 48.33%. Gold is driven by entirely different macro forces, so weather-linked features don't help here.
- **Wheat**: Buy & Hold (51.67%) remains the highest, leaving the model slightly behind.
- **Natural Gas**: Persistence wins at 53.33%, showing stronger momentum than what the model extracts.

**Regime Breakdown:**
- For soybean, accuracy rises above 60% during the Post-COVID Inflation regime.
- For cocoa, accuracy drops during the 2023–2026 price spikes, suggesting the historical ONI/price relationship shifted during that period.

## Conclusion

Combining climate indices with macro indicators works selectively. It adds measurable predictive value in weather-sensitive soft commodities like soybeans, but fails to beat simple trend or momentum baselines for assets like gold, wheat, or natural gas. 

## Limitations & Future Extensions

- **No statistical checks**: We didn't test if the hit ratios are actually meaningful or just random noise.
- **No backtest**: This project only measures directional accuracy, it is not a full trading strategy simulation.
- **Simulation & Equity Curves**: Moving beyond directional accuracy to simulate a full trading strategy, plotting cumulative equity curves (portfolio wealth over time) versus Buy & Hold benchmarks, possibly including transaction costs and maximum drawdown metrics.
- **Fixed hyperparameters**: Used a single default configuration for XGBoost without grid search.
- **Manual regimes**: Macro regimes were hand-picked rather than detected algorithmically.