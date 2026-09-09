import io

import pandas as pd
import requests
import yfinance as yf


def fetch_oni_data(fetch_func=requests.get) -> pd.DataFrame:
    """Download and parse NOAA's Oceanic Niño Index into a monthly Date/ONI DataFrame."""
    url = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
    response = fetch_func(url)

    if response.status_code != 200:
        raise ConnectionError("Failed to fetch NOAA ONI data.")

    df = pd.read_csv(io.StringIO(response.text), sep=r"\s+")

    month_map = {
        "DJF": 1,
        "JFM": 2,
        "FMA": 3,
        "MAM": 4,
        "AMJ": 5,
        "MJJ": 6,
        "JJA": 7,
        "JAS": 8,
        "ASO": 9,
        "SON": 10,
        "OND": 11,
        "NDJ": 12,
    }
    df["Month_Num"] = df["SEAS"].map(month_map)

    df["Date"] = pd.to_datetime(
        df["YR"].astype(str) + "-" + df["Month_Num"].astype(str).str.zfill(2) + "-01"
    )

    df = df[["Date", "ANOM"]].rename(columns={"ANOM": "ONI"})
    return df.sort_values("Date").reset_index(drop=True)


def fetch_commodity_data(
    ticker_mapping: dict | None = None,
    start_date: str = "2000-09-01",
    download_func=yf.download,
) -> pd.DataFrame:
    """Download monthly closing prices for the mapped tickers and rename columns to readable labels."""
    if ticker_mapping is None:
        ticker_mapping = {
            "KC=F": "coffee_price",
            "CC=F": "cocoa_price",
            "ZS=F": "soybean_price",
            "ZW=F": "wheat_price",
            "NG=F": "natural_gas_price",
            "GC=F": "gold_price",
            "^IRX": "us_interest_rate",
            "EURUSD=X": "usd_index",
        }

    tickers = list(ticker_mapping.keys())
    raw_data = download_func(tickers, start="2000-01-01")["Close"]

    renamed_data = raw_data.rename(columns=ticker_mapping)

    if "usd_index" in renamed_data.columns:
        renamed_data["usd_index"] = 1 / renamed_data["usd_index"]

    monthly_data = renamed_data.resample("ME").last()
    monthly_data.index = monthly_data.index.to_period("M").to_timestamp()
    monthly_data.index.name = "Date"

    monthly_data = monthly_data.bfill()
    monthly_data = monthly_data[monthly_data.index >= start_date]

    return monthly_data


def load_merged_dataset(
    oni_df: pd.DataFrame | None = None,
    commodities_df: pd.DataFrame | None = None,
    start_date: str = "2000-09-01",
) -> pd.DataFrame:
    """Fetch (or accept) ONI and commodity data and inner-join them on Date into one monthly DataFrame."""
    if oni_df is None:
        oni_df = fetch_oni_data()
    if commodities_df is None:
        commodities_df = fetch_commodity_data(start_date=start_date)

    oni_df = oni_df[oni_df["Date"] >= start_date]
    merged_df = pd.merge(oni_df, commodities_df, on="Date", how="inner")

    return merged_df.set_index("Date")
