import pandas as pd
import numpy as np

def to_daily_pnl(pnl_ts, start=None, end=None, freq='B'):
    """
    Convert sparse MTM series (only in-trade days) into a continuous daily series.
    Days without a position => equity is flat => daily return = 0.
    """
    s = pnl_ts["MTM"].copy()
    s.index = pd.to_datetime(s.index)

    if start is None:
        start = s.index.min()
    if end is None:
        end = s.index.max()

    # business-day calendar
    cal = pd.bdate_range(start=start, end=end, freq=freq)

    # equity = cumulative PnL; forward-fill across gaps; before first value -> 0
    equity = s.cumsum().reindex(cal).ffill().fillna(0.0)

    # daily PnL increments (returns); gaps become 0 because equity is flat
    daily_pnl = equity.diff().fillna(0.0)

    # return both for convenience
    return pd.DataFrame({"Equity": equity, "DailyPnL": daily_pnl})

import numpy as np

def compute_sharpe_from_daily(daily_df, freq=252):
    r = daily_df["DailyPnL"]
    std = r.std()
    return (r.mean() / std) * np.sqrt(freq) if std != 0 else np.nan

def compute_metrics(trades_df, pnl_ts, start=None, end=None):
    daily = to_daily_pnl(pnl_ts, start=start, end=end)
    sharpe = compute_sharpe_from_daily(daily)
    max_dd = (daily["Equity"] - daily["Equity"].cummax()).min()
    return {
        "Sharpe": sharpe,
        "Cumulative PnL": float(daily["Equity"].iloc[-1]),
        "Win Rate": (trades_df["PnL"] > 0).mean() if not trades_df.empty else np.nan,
        "Average Trade PnL": trades_df["PnL"].mean() if not trades_df.empty else np.nan,
        "Max Drawdown": float(max_dd),
    }


import matplotlib.pyplot as plt

def plot_results(pnl_ts, trades_df, start=None, end=None):
    daily = to_daily_pnl(pnl_ts, start=start, end=end)

    plt.figure(figsize=(10,5))
    daily["Equity"].plot(title="Equity Curve (business days)")
    plt.ylabel("Cumulative PnL"); plt.xlabel("")
    plt.tight_layout(); plt.show()

    if not trades_df.empty:
        plt.figure(figsize=(8,4))
        trades_df["PnL"].hist(bins=20)
        plt.title("Distribution of Trade PnLs"); plt.xlabel("PnL per trade")
        plt.tight_layout(); plt.show()


-----
import matplotlib.pyplot as plt
import seaborn as sns

# define tenor buckets
tenor_bins_bd = [0,21,63,126,252,504,756]
tenor_labels = ['<1M','1–3M','3–6M','6–12M','1–2Y','2–3Y']

# filter only single legs and straddle
df = df[(df["NUMBER_OF_LEGS"] == 1) | (df["STRATEGY_TYPE"] == "Straddle")]

# use your existing strike bin column
df["_strike_bucket"] = df["STRIKE_PCT_SPOT_BIN"]
df["_tenor_bucket"]  = pd.cut(
    df["MATURITY_in_BDAYS"],
    bins=tenor_bins_bd,
    labels=tenor_labels,
    right=False
)

# loop over strategies
for strat in df["STRATEGY_TYPE"].unique():
    dd = df[df["STRATEGY_TYPE"] == strat]

    # pivot table of notional
    heat = (dd.groupby(["_tenor_bucket","_strike_bucket"])["NOTIONAL_USD"]
              .sum()
              .unstack(fill_value=0))

    # force column order to match your "labels" list
    heat = heat.reindex(columns=labels, fill_value=0)

    # normalize to %
    heat = heat / heat.values.sum()

    # annotations in %
    annot = (heat*100).round(0).astype(int).astype(str) + "%"

    # plot
    plt.figure(figsize=(14,6))
    sns.heatmap(
        heat, cmap="Blues", vmin=0, vmax=1,
        annot=annot, fmt="", annot_kws={"fontsize":10, "color":"black"},
        cbar_kws={"label":"% of strategy notional"}
    )
    plt.title(f"Strike% × Tenor – {strat}", fontsize=16, pad=10)
    plt.xlabel("Strike% bucket")
    plt.ylabel("Tenor bucket")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
