"""
Tests for add_final_22_missing_columns.
Run: python -m pytest test_final_22.py -v
      or: python test_final_22.py
"""
import numpy as np
import pandas as pd

from column_library import add_final_22_missing_columns, get_column_groups

FINAL_22_NAMES = [
    "Col_ALMA_10_6_0.85",
    "Col_FWMA_10",
    "Col_HWMA_10",
    "Col_PWMA_10",
    "Col_ZLMA_10",
    "Col_SuperTrend",
    "Col_SuperTrend_Direction",
    "Col_Keltner_Upper",
    "Col_Keltner_Lower",
    "Col_Keltner_Width_ATR",
    "Col_Donchian_Upper",
    "Col_Donchian_Lower",
    "Col_MassIndex_25",
    "Col_UlcerIndex_14",
    "Col_ChoppinessIndex_14",
    "Col_ChaikinMoneyFlow_20",
    "Col_ElderForceIndex_13",
    "Col_EaseOfMovement_14",
    "Col_NVI",
    "Col_PVI",
    "Col_PSAR",
    "Col_QStick_10",
    "Col_RelativeVigorIndex",
    "Col_RVISignal",
    "Col_SchaffTrendCycle",
]


def test_final_22_single_ticker():
    """Single-ticker with DatetimeIndex, Open/High/Low/Close/Volume."""
    np.random.seed(42)
    n = 250
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
    out = add_final_22_missing_columns(df)
    for col in FINAL_22_NAMES:
        assert col in out.columns, f"Missing {col}"
    assert len([c for c in out.columns if c.startswith("Col_")]) >= 22


def test_final_22_long_format():
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
    out = add_final_22_missing_columns(df)
    for col in FINAL_22_NAMES:
        assert col in out.columns, f"Missing {col}"
    assert "Ticker" in out.columns


def test_final_22_registry():
    """final_22 group in get_column_groups contains all 22 names."""
    groups = get_column_groups()
    assert "final_22" in groups
    reg = set(groups["final_22"])
    for col in FINAL_22_NAMES:
        assert col in reg, f"Registry missing {col}"


if __name__ == "__main__":
    test_final_22_single_ticker()
    test_final_22_long_format()
    test_final_22_registry()
    print("All tests passed.")
