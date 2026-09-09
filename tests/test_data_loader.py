import pandas as pd
import pytest

from src.data_loader import fetch_commodity_data, fetch_oni_data, load_merged_dataset


@pytest.fixture
def mock_noaa_response():
    """Return a small mock NOAA ONI response body for parsing tests."""
    return """ SEAS   YR   TOTAL   ANOM
DJF  2020   27.2    0.5
JFM  2020   27.1    0.3
FMA  2020   26.9    0.1
"""


def test_fetch_oni_data_success(mock_noaa_response):
    """A 200 response should parse into a two-column Date/ONI DataFrame."""

    class MockResponse:
        status_code = 200
        text = mock_noaa_response

    df_oni = fetch_oni_data(fetch_func=lambda url: MockResponse())

    assert list(df_oni.columns) == ["Date", "ONI"]
    assert len(df_oni) == 3
    assert df_oni["ONI"].iloc[0] == 0.5


def test_fetch_oni_data_failure():
    """A non-200 response should raise ConnectionError."""

    class MockResponse:
        status_code = 404

    with pytest.raises(ConnectionError):
        fetch_oni_data(fetch_func=lambda url: MockResponse())


def test_fetch_commodity_data():
    """Ticker prices should be renamed to the mapped commodity columns."""
    dates = pd.date_range(start="2020-01-01", periods=3, freq="MS", name="Date")
    mock_df = pd.DataFrame(
        {
            "CC=F": [2500, 2600, 2700],
            "GC=F": [1500, 1550, 1600],
            "^IRX": [1.2, 1.3, 1.4],
            "EURUSD=X": [1.1, 1.09, 1.08],
        },
        index=dates,
    )

    mapping = {
        "CC=F": "cocoa_price",
        "GC=F": "gold_price",
        "^IRX": "us_interest_rate",
        "EURUSD=X": "usd_index",
    }

    df_commodities = fetch_commodity_data(
        ticker_mapping=mapping,
        download_func=lambda tickers, start: {"Close": mock_df},
    )

    assert "cocoa_price" in df_commodities.columns
    assert "gold_price" in df_commodities.columns
    assert "us_interest_rate" in df_commodities.columns
    assert "usd_index" in df_commodities.columns
    assert len(df_commodities) == 3


def test_load_merged_dataset():
    """ONI and commodity data should inner-join into one Date-indexed DataFrame."""
    dates = pd.date_range(start="2020-01-01", periods=2, freq="MS", name="Date")

    mock_oni = pd.DataFrame({"Date": dates, "ONI": [0.5, 0.3]})
    mock_commodities = pd.DataFrame(
        {"cocoa_price": [2500, 2600], "usd_index": [95.0, 96.0]}, index=dates
    )

    df_merged = load_merged_dataset(oni_df=mock_oni, commodities_df=mock_commodities)

    assert df_merged.index.name == "Date"
    assert "ONI" in df_merged.columns
    assert "cocoa_price" in df_merged.columns
    assert "usd_index" in df_merged.columns
    assert len(df_merged) == 2
