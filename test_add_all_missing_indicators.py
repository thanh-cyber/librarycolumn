"""
Quick test for add_all_missing_indicators.
Run: python -m pytest test_add_all_missing_indicators.py -v
       or: python test_add_all_missing_indicators.py
"""
import numpy as np
import pandas as pd

from column_library import add_all_missing_indicators


def test_single_ticker_capitalized_ohlcv():
    """Single-ticker with DatetimeIndex, Open/High/Low/Close/Volume."""
    np.random.seed(42)
    n = 500
    idx = pd.date_range("2024-01-01", periods=n, freq="1min")
    df = pd.DataFrame(
        {
            "Open": 100 + np.cumsum(np.random.randn(n) * 0.1),
            "High": 101 + np.cumsum(np.random.randn(n) * 0.1),
            "Low": 99 + np.cumsum(np.random.randn(n) * 0.1),
            "Close": 100 + np.cumsum(np.random.randn(n) * 0.1),
            "Volume": np.random.randint(1000, 10000, n),
        },
        index=idx,
    )
    df["High"] = df[["Open", "High", "Close"]].max(axis=1)
    df["Low"] = df[["Open", "Low", "Close"]].min(axis=1)
    out = add_all_missing_indicators(df)
    cols = [c for c in out.columns if c.startswith("Col_")]
    assert len(cols) >= 70, f"Expected >= 70 new Col_* columns, got {len(cols)}"
    assert "Col_DEMA20" in out.columns
    assert "Col_HeikinAshi_Open" in out.columns


def test_long_format_lowercase_ohlcv():
    """Long format with Ticker, datetime, open/high/low/close/volume."""
    np.random.seed(42)
    n = 300
    times = pd.date_range("2024-01-01 09:30", periods=n, freq="1min")
    df = pd.DataFrame(
        {
            "Ticker": ["AAPL"] * n,
            "datetime": list(times),
            "open": 100 + np.cumsum(np.random.randn(n) * 0.1),
            "high": 101 + np.cumsum(np.random.randn(n) * 0.1),
            "low": 99 + np.cumsum(np.random.randn(n) * 0.1),
            "close": 100 + np.cumsum(np.random.randn(n) * 0.1),
            "volume": np.random.randint(1000, 10000, n),
        }
    )
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)
    out = add_all_missing_indicators(df)
    cols = [c for c in out.columns if c.startswith("Col_")]
    assert len(cols) >= 70
    assert "Col_HeikinAshi_Open" in out.columns
    assert "Ticker" in out.columns


if __name__ == "__main__":
    test_single_ticker_capitalized_ohlcv()
    test_long_format_lowercase_ohlcv()
    print("All tests passed.")
