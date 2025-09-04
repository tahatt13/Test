import pandas as pd
import numpy as np

# choose the order from your lookback_periods dict (e.g., '1m','2m',...,'12m')
labels = list(lookback_periods.keys())

metrics_cols = ["Sharpe", "Cumulative PnL", "Win Rate", "Average Trade PnL", "Max Drawdown"]
assets = {
    "ASML": ASML_metrics,
    "BESI": BESI_metrics,
    "INTC": INTC_metrics,
}

# build a single table
rows = {}
for lbl in labels:
    row = {}
    for asset, md in assets.items():
        m = md.get(lbl, {})
        for k in metrics_cols:
            row[(asset, k)] = m.get(k, np.nan)
    rows[lbl] = row

metrics_table = pd.DataFrame.from_dict(rows, orient="index")
metrics_table.columns = pd.MultiIndex.from_tuples(metrics_table.columns, names=["Asset", "Metric"])

# format a pretty print (win rate in %, rounded)
pretty = metrics_table.copy()
for asset in assets:
    col = (asset, "Win Rate")
    if col in pretty.columns:
        pretty[col] = (pretty[col] * 100)

# rounding per metric
round_map = {}
for asset in assets:
    round_map[(asset, "Sharpe")] = 2
    round_map[(asset, "Cumulative PnL")] = 2
    round_map[(asset, "Average Trade PnL")] = 4
    round_map[(asset, "Max Drawdown")] = 2
    round_map[(asset, "Win Rate")] = 1

with pd.option_context("display.max_columns", None, "display.width", 220):
    print(pretty.round(round_map))
