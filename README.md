import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# start from your result
g = (cluster_16
     .groupby(['STRATEGY_TYPE','UNDERLYING_INSTRUMENT_NAME'])['NOTIONAL_USD']
     .sum().reset_index())

# total notional per strategy (for sorting and “Other”)
tot = g.groupby('STRATEGY_TYPE')['NOTIONAL_USD'].sum().rename('TOTAL')
g = g.merge(tot, on='STRATEGY_TYPE')
g['share'] = g['NOTIONAL_USD'] / g['TOTAL']

# keep top-3 underlyings per strategy, aggregate the rest as "Other"
g_sorted = g.sort_values(['STRATEGY_TYPE','NOTIONAL_USD'], ascending=[True,False])
top3 = g_sorted.groupby('STRATEGY_TYPE').head(3)
other = (g_sorted.groupby('STRATEGY_TYPE').tail(len(g_sorted))
         .groupby('STRATEGY_TYPE')
         .apply(lambda x: 1 - x.nlargest(3, 'NOTIONAL_USD')['share'].sum())
         .rename('share').reset_index())
other['UNDERLYING_INSTRUMENT_NAME'] = 'Other'

plotdf = pd.concat([
    top3[['STRATEGY_TYPE','UNDERLYING_INSTRUMENT_NAME','share']],
    other[['STRATEGY_TYPE','UNDERLYING_INSTRUMENT_NAME','share']]
], ignore_index=True)

# order strategies by total notional
order = tot.sort_values(ascending=False).index.tolist()
plotdf['STRATEGY_TYPE'] = pd.Categorical(plotdf['STRATEGY_TYPE'], order)

# pivot to stacked 100% bars
wide = (plotdf.pivot(index='STRATEGY_TYPE',
                     columns='UNDERLYING_INSTRUMENT_NAME',
                     values='share')
        .fillna(0).sort_index())

# plot
fig, ax = plt.subplots(figsize=(12, 7))
left = np.zeros(len(wide))
for col in wide.columns:
    ax.barh(wide.index.astype(str), wide[col].values, left=left, label=col)
    left += wide[col].values

# labels (% on segments that matter)
for i, y in enumerate(wide.index):
    cum = 0.0
    for col in wide.columns:
        val = wide.loc[y, col]
        if val >= 0.08:  # show only meaningful pieces
            ax.text(cum + val/2, i, f"{col} • {val:.0%}",
                    ha='center', va='center', fontsize=9, color='white')
        cum += val

ax.set_title("Top Underlyings per Strategy — Share of Strategy Notional (100% = each strategy)", fontsize=14)
ax.set_xlabel("Share of strategy notional")
ax.set_xlim(0, 1)
ax.legend(title="Underlying", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()




------
import seaborn as sns
import matplotlib.pyplot as plt

# Suppose you already have `top3_notional` with columns:
# STRATEGY_TYPE, UNDERLYING_INSTRUMENT_NAME, total_notional

plt.figure(figsize=(12,6))
sns.barplot(data=top3_notional,
            x="total_notional", y="STRATEGY_TYPE",
            hue="UNDERLYING_INSTRUMENT_NAME", orient="h")

plt.title("Top 3 Underlyings by Strategy (Notional)", fontsize=14)
plt.xlabel("Total Notional (USD)")
plt.ylabel("Strategy")
plt.legend(title="Underlying", bbox_to_anchor=(1.05,1), loc="upper left")
plt.tight_layout()
plt.show()

--------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def strategy_heatmaps(
    df,
    cluster_id=None,                 # e.g. 16 to filter one cluster; or None for all
    strategies=None,                 # list of STRATEGY_TYPE to show; None = all present
    strike_bins=np.arange(60, 141, 5),          # 60%..140% by 5%
    tenor_bins_bd=[0,21,63,126,252,504,756],    # <1M, 1–3M, 3–6M, 6–12M, 1–2Y, 2–3Y
    tenor_labels=['<1M','1–3M','3–6M','6–12M','1–2Y','2–3Y'],
    normalize=False,                 # True → each heatmap is % of strategy notional
    min_cell_value=0.0               # set >0 to mask tiny cells (e.g. 0.01 for 1% when normalize=True)
):
    d = df.copy()

    # Optional cluster filter
    if cluster_id is not None and 'cluster_kmeans' in d.columns:
        d = d[d['cluster_kmeans'] == cluster_id].copy()

    # ensure numeric
    for c in ['NOTIONAL_USD','STRIKE_PCT','STRIKE_2','MATURITY_in_BDAYS','MATURITY_WIDTH']:
        d[c] = pd.to_numeric(d[c], errors='coerce')

    # midpoint per trade (single-leg vs multi-leg)
    is_single = d['NUMBER_OF_LEGS'] == 1
    d['_strike_mid'] = np.where(is_single, d['STRIKE_PCT'],
                                (d['STRIKE_PCT'] + d['STRIKE_2'])/2.0)
    d['_mat_mid_bd'] = np.where(is_single, d['MATURITY_in_BDAYS'],
                                d['MATURITY_in_BDAYS'] + d['MATURITY_WIDTH']/2.0)

    # bucketize
    d['_strike_bucket'] = pd.cut(d['_strike_mid'], bins=strike_bins, include_lowest=True)
    d['_tenor_bucket']  = pd.cut(d['_mat_mid_bd'], bins=tenor_bins_bd,
                                 labels=tenor_labels, include_lowest=True)

    # which strategies to plot
    all_strats = d['STRATEGY_TYPE'].dropna().unique().tolist()
    if strategies is None:
        strategies = all_strats
    else:
        strategies = [s for s in strategies if s in all_strats]
    if not strategies:
        print("No matching STRATEGY_TYPE to plot."); return

    # layout
    n = len(strategies)
    cols = 3
    rows = int(np.ceil(n/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 3.8*rows), squeeze=False)

    vmax_global = None
    if not normalize:
        # For a consistent color scale across strategies, compute a global max
        tmp = (d.groupby(['STRATEGY_TYPE','_tenor_bucket','_strike_bucket'])['NOTIONAL_USD']
                 .sum().reset_index())
        vmax_global = tmp['NOTIONAL_USD'].max()

    for i, strat in enumerate(strategies):
        ax = axes[i//cols, i%cols]
        dd = d[d['STRATEGY_TYPE'] == strat].dropna(subset=['_strike_bucket','_tenor_bucket','NOTIONAL_USD'])
        if dd.empty:
            ax.axis('off'); ax.set_title(f"{strat} (no data)"); continue

        heat = (dd.groupby(['_tenor_bucket','_strike_bucket'])['NOTIONAL_USD']
                  .sum().unstack(fill_value=0))

        if normalize:
            denom = heat.values.sum()
            if denom > 0:
                heat = heat / denom
            cbar_label = '% of strategy notional'
            # optional small-value masking
            if min_cell_value > 0:
                heat = heat.mask(heat < min_cell_value)
            vmax = 1.0
            fmt = '.0%' if heat.max().max() <= 1 else '.2f'
        else:
            cbar_label = 'Total notional (USD)'
            if min_cell_value > 0:
                heat = heat.mask(heat < min_cell_value)
            vmax = vmax_global
            fmt = '.0f'

        sns.heatmap(heat, ax=ax, cmap='Blues', cbar_kws={'label': cbar_label},
                    vmin=0, vmax=vmax)
        ax.set_title(strat)
        ax.set_xlabel('Strike% bucket')
        ax.set_ylabel('Tenor bucket')

    # hide any empty subplots
    for j in range(i+1, rows*cols):
        axes[j//cols, j%cols].axis('off')

    suptitle = "Strike% × Tenor – Notional concentration per strategy"
    if cluster_id is not None:
        suptitle += f" (Cluster {cluster_id})"
    if normalize:
        suptitle += " – normalized"
    fig.suptitle(suptitle, fontsize=14)
    plt.tight_layout(rect=[0,0,1,0.96])
    plt.show()

# --- Example calls ---
# Custom strike bins (your version)
bins_strike = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 96, 97, 98, 99, 100,
               101, 102, 103, 104, 105, 110, 120, np.inf]

labels_strike = ['0-10%', '10-20%', '20-30%', '30-40%', '40-50%', '50-60%',
                 '60-70%', '70-80%', '80-90%', '90-95%',
                 '95-96%', '96-97%', '97-98%', '98-99%',
                 '99-100%', '100-101%', '101-102%', '102-103%',
                 '103-104%', '104-105%', '105-110%', '110-120%', '120%+']

# Pass into function
strategy_heatmaps(
    df,
    cluster_id=16,                             # or None for all clusters
    strategies=['European Call','European Put','Bull Call Spread'],
    strike_bins=bins_strike,
    normalize=True,
    min_cell_value=0.01                        # hide cells <1% of flow
)
