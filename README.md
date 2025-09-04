import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 1) PRICE + SMAs + ENTRY/EXIT MARKERS ------------------------------------
def plot_price_with_trades(symbol, lookback_label, piv_data, trades_df,
                           sma_short=50, sma_long=200, title_suffix=""):
    s = piv_data[symbol].dropna().copy()
    sma_s = s.rolling(sma_short).mean()
    sma_l = s.rolling(sma_long).mean()

    plt.figure(figsize=(11, 4))
    s.plot(label="Price")
    sma_s.plot(label=f"SMA{sma_short}")
    sma_l.plot(label=f"SMA{sma_long}")

    # entry / exit markers
    if trades_df is not None and not trades_df.empty:
        ent = pd.to_datetime(trades_df["entry"])
        exi = pd.to_datetime(trades_df["exit"])
        y_ent = s.reindex(ent, method="nearest")
        y_exi = s.reindex(exi, method="nearest")
        plt.scatter(y_ent.index, y_ent.values, marker="^", label="Entry")
        plt.scatter(y_exi.index, y_exi.values, marker="v", label="Exit")

        # optional: shade in-trade windows
        for a, b in zip(ent, exi):
            plt.axvspan(a, b, alpha=0.1)

    plt.title(f"{symbol} | Price + SMA filter | lookback={lookback_label}{title_suffix}")
    plt.legend()
    plt.tight_layout()
    plt.show()

# --- 2) EQUITY CURVE ----------------------------------------------------------
def plot_equity_curve(pnl_ts, start=None, end=None, title="Equity curve"):
    daily = to_daily_pnl(pnl_ts, start=start, end=end)  # uses your function
    plt.figure(figsize=(10, 4))
    daily["Equity"].plot(title=title)
    plt.ylabel("Cumulative PnL")
    plt.tight_layout()
    plt.show()

# --- 3) TRADE PnL HISTOGRAM ---------------------------------------------------
def plot_trade_pnl_hist(trades_df, pct=True, title="Distribution of trade PnLs"):
    if trades_df is None or trades_df.empty:
        print("No trades to plot."); return
    col = "PnL_pct" if pct else "PnL_abs"
    plt.figure(figsize=(8, 4))
    trades_df[col].hist(bins=30)
    plt.title(title)
    plt.xlabel(col)
    plt.tight_layout()
    plt.show()

# --- 4) COVERAGE (trade count) vs PERFORMANCE (Sharpe) ------------------------
def plot_trades_vs_sharpe(metrics_dict, trades_dict_for_asset, asset_label):
    # metrics_dict: e.g., ASML_metrics  (mapping lookback_label -> metrics dict)
    # trades_dict_for_asset: mapping lookback_label -> list of (entry, exit)
    rows = []
    for lbl, m in metrics_dict.items():
        n_tr = len(trades_dict_for_asset.get(lbl, []))
        sh = m.get("Sharpe", np.nan)
        rows.append((lbl, n_tr, sh))
    df = pd.DataFrame(rows, columns=["lookback", "n_trades", "Sharpe"]).set_index("lookback")

    # bar for counts + line for Sharpe (two separate figures to keep it simple)
    plt.figure(figsize=(9, 3.8))
    df["n_trades"].plot(kind="bar", title=f"{asset_label} – Number of trades by lookback")
    plt.tight_layout(); plt.show()

    plt.figure(figsize=(9, 3.8))
    df["Sharpe"].plot(marker="o", title=f"{asset_label} – Sharpe by lookback")
    plt.tight_layout(); plt.show()

# --- 5) OPTIONAL: Lookback sensitivity table → line plot ----------------------
def plot_sharpe_sensitivity(asml_m, besi_m, intc_m):
    def grab(mdict):
        return pd.Series({k: v.get("Sharpe", np.nan) for k, v in mdict.items()}).sort_index()
    plt.figure(figsize=(10, 4))
    grab(asml_m).plot(marker="o", label="ASML")
    grab(besi_m).plot(marker="o", label="BESI")
    grab(intc_m).plot(marker="o", label="INTC")
    plt.title("Sharpe vs lookback")
    plt.legend(); plt.tight_layout(); plt.show()



lbl = "12m"  # <- change as needed

# You already computed these somewhere:
# ASML_trades_df, ASML_pnl_ts = compute_trade_pnls(ASML_pnls, trades_dict[lbl]["ASML.AS"])
# BESI_trades_df, BESI_pnl_ts = ...
# INTC_trades_df, INTC_pnl_ts = ...

# 1) Price + SMAs + trades
plot_price_with_trades("ASML.AS", lbl, piv_data, ASML_trades_df)
plot_price_with_trades("BESI.AS", lbl, piv_data, BESI_trades_df)
plot_price_with_trades("INTC.OQ", lbl, piv_data, INTC_trades_df)

# 2) Equity curves
plot_equity_curve(ASML_pnl_ts, title=f"ASML {lbl} – equity")
plot_equity_curve(BESI_pnl_ts, title=f"BESI {lbl} – equity")
plot_equity_curve(INTC_pnl_ts, title=f"INTC {lbl} – equity")

# 3) Trade PnL histograms (% of premium)
plot_trade_pnl_hist(ASML_trades_df, pct=True, title=f"ASML {lbl} – trade PnLs (%)")
plot_trade_pnl_hist(BESI_trades_df, pct=True, title=f"BESI {lbl} – trade PnLs (%)")
plot_trade_pnl_hist(INTC_trades_df, pct=True, title=f"INTC {lbl} – trade PnLs (%)")

# 4) Coverage vs performance across lookbacks
plot_trades_vs_sharpe(ASML_metrics, trades_dict={k: v["ASML.AS"] for k, v in trades_dict.items()}, asset_label="ASML")
plot_trades_vs_sharpe(BESI_metrics, trades_dict={k: v["BESI.AS"] for k, v in trades_dict.items()}, asset_label="BESI")
plot_trades_vs_sharpe(INTC_metrics, trades_dict={k: v["INTC.OQ"] for k, v in trades_dict.items()}, asset_label="INTC")

# 5) Optional sensitivity
plot_sharpe_sensitivity(ASML_metrics, BESI_metrics, INTC_metrics)
