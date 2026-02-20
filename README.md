# librarycolumn

Portable, reusable, single-file OHLCV entry-feature library for backtesting.

Install once, import anywhere:

```python
from column_library import add_all_columns, get_all_column_names, get_column_groups
```

## Install

### From GitHub

```bash
pip install "git+https://github.com/thanh-cyber/librarycolumn.git"
```

### Editable (local development)

```bash
git clone https://github.com/thanh-cyber/librarycolumn.git
cd librarycolumn
pip install -e .
```

## Quick Usage

```python
import pandas as pd
from column_library import add_all_columns, add_volume_vwap_columns, add_volatility_columns

df = pd.read_csv("ohlcv.csv", parse_dates=["timestamp"], index_col="timestamp")

# Add all columns
df_all = add_all_columns(df)

# Add only specific categories
df_some = add_volatility_columns(df.copy())
df_some = add_volume_vwap_columns(df_some, inplace=True)
```

## Data requirements

- Columns: `Open`, `High`, `Low`, `Close`, `Volume`
- Index: sorted `pandas.DatetimeIndex` (daily or intraday)

## Public API

- `add_all_columns(df, inplace=False)`
- `add_volatility_columns(df, inplace=False)`
- `add_trend_momentum_columns(df, inplace=False)`
- `add_oscillator_columns(df, inplace=False)`
- `add_volume_vwap_columns(df, inplace=False)`
- `add_price_action_columns(df, inplace=False)`
- `add_gaps_columns(df, inplace=False)`
- `add_key_levels_columns(df, inplace=False)`
- `add_market_context_columns(df, inplace=False)`
- `add_time_columns(df, inplace=False)`
- `get_all_column_names()`
- `get_column_groups()`
