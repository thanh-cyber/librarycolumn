"""
Tests for add_cruncher_context_columns.
Run: python -m pytest test_cruncher_context.py -v
      or: python test_cruncher_context.py
"""
import numpy as np
import pandas as pd

from column_library import add_cruncher_context_columns, get_column_groups

# cruncher_context registry (Col_Gap_Pct lives in "gaps", not duplicated here)
CRUNCHER_COLS = [
    "Col_GapFill_15min_Pct",
    "Col_GapFill_15min_Zscore",
    "Col_ORB_5min_BreakHigh",
    "Col_ORB_5min_BreakLow",
    "Col_ORB_5min_DistHigh_ATR",
    "Col_ORB_15min_BreakHigh",
    "Col_ORB_15min_BreakLow",
    "Col_ORB_15min_DistHigh_ATR",
    "Col_ORB_30min_BreakHigh",
    "Col_ORB_30min_BreakLow",
    "Col_ORB_30min_DistHigh_ATR",
    "Col_ORB_60min_BreakHigh",
    "Col_ORB_60min_BreakLow",
    "Col_ORB_60min_DistHigh_ATR",
    "Col_ExtensionFromDaily9EMA_ATR",
    "Col_ExtensionFromDaily9EMA_Rank",
    "Col_MultiDaySlope_5d",
    "Col_InsideDay",
    "Col_DollarVolume_20dAvg",
    "Col_RelativeVolume_First30min",
    "Col_RelativeVolume_First30min_Rank",
    "Col_VolumeSurge_1min_Ratio",
    "Col_IntradayATR_Ratio",
    "Col_StdDev_Last10Bars_ATR",
    "Col_GapPct_x_RelVol30",
    "Col_GapPct_x_ExtensionATR",
]


def test_cruncher_single_ticker():
    """Single-ticker with DatetimeIndex, Open/High/Low/Close/Volume."""
    np.random.seed(42)
    n = 250
    idx = pd.date_range("2024-01-01 09:30", periods=n, freq="1min")
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
    out = add_cruncher_context_columns(df)
    assert "Col_Gap_Pct" in out.columns
    for col in CRUNCHER_COLS:
        assert col in out.columns, f"Missing {col}"


def test_cruncher_long_format():
    """Long format with Ticker, datetime, open/high/low/close/volume."""
    np.random.seed(43)
    n = 250
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
    out = add_cruncher_context_columns(df)
    assert "Ticker" in out.columns
    assert "Col_Gap_Pct" in out.columns
    for col in CRUNCHER_COLS:
        assert col in out.columns, f"Missing {col}"


def test_cruncher_registry():
    """cruncher_context group in get_column_groups matches expected list."""
    groups = get_column_groups()
    assert "cruncher_context" in groups
    reg = set(groups["cruncher_context"])
    for col in CRUNCHER_COLS:
        assert col in reg, f"Registry missing {col}"
    # Col_Gap_Pct should not be duplicated in cruncher_context (lives in gaps)
    assert "Col_Gap_Pct" not in reg


if __name__ == "__main__":
    test_cruncher_single_ticker()
    test_cruncher_long_format()
    test_cruncher_registry()
    print("All tests passed.")
