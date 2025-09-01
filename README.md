import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def strategy_heatmaps_one_per_line(
    df,
    cluster_id=None,
    strategies=None,
    top_strategies=6,
    strike_bins=None,
    strike_labels=None,
    tenor_bins_bd=[0,21,63,126,252,504,756],
    tenor_labels=['<1M','1–3M','3–6M','6–12M','1–2Y','2–3Y'],
    normalize=True,
    label_min_pct=0.02,   # cells below this threshold shown as 0%
    figsize_per=(10,4)
):
    d = df.copy()

    # optional cluster filter
    if cluster_id is not None and 'cluster_kmeans' in d.columns:
        d = d[d['cluster_kmeans'] == cluster_id].copy()

    # ensure numeric
    for c in ['NOTIONAL_USD','STRIKE_PCT','STRIKE_2','MATURITY_in_BDAYS','MATURITY_WIDTH','NUMBER_OF_LEGS']:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors='coerce')

    # midpoints
    is_single = (d['NUMBER_OF_LEGS'] == 1)
    d['_strike_mid'] = np.where(is_single, d['STRIKE_PCT'],
                                (d['STRIKE_PCT'] + d['STRIKE_2'])/2.0)
    d['_mat_mid_bd'] = np.where(is_single, d['MATURITY_in_BDAYS'],
                                d['MATURITY_in_BDAYS'] + d['MATURITY_WIDTH']/2.0)

    d = d.dropna(subset=['_strike_mid','_mat_mid_bd','NOTIONAL_USD','STRATEGY_TYPE'])

    # strike bins (focus around ATM if not provided)
    if strike_bins is None:
        strike_bins = np.arange(90, 111, 2)
    d['_strike_bucket'] = pd.cut(d['_strike_mid'], bins=strike_bins, include_lowest=True,
                                 labels=strike_labels if strike_labels else None, right=False)
    d['_tenor_bucket']  = pd.cut(d['_mat_mid_bd'], bins=tenor_bins_bd, include_lowest=True,
                                 labels=tenor_labels, right=False)

    # pick strategies
    if strategies is None:
        strat_order = (d.groupby('STRATEGY_TYPE')['NOTIONAL_USD']
                         .sum().sort_values(ascending=False).index.tolist())
        strategies = strat_order[:top_strategies]
    else:
        have = set(d['STRATEGY_TYPE'].unique())
        strategies = [s for s in strategies if s in have]

    if not strategies:
        print("No matching STRATEGY_TYPE to plot."); return

    # --- PLOT ---
    fig, axes = plt.subplots(len(strategies), 1,
                             figsize=(figsize_per[0], figsize_per[1]*len(strategies)),
                             squeeze=False)

    vmax_abs = None
    if not normalize:
        vmax_abs = d.groupby(['STRATEGY_TYPE','_tenor_bucket','_strike_bucket'])['NOTIONAL_USD'].sum().max()

    for i, strat in enumerate(strategies):
        ax = axes[i,0]
        dd = d[d['STRATEGY_TYPE'] == strat]

        heat = (dd.groupby(['_tenor_bucket','_strike_bucket'])['NOTIONAL_USD']
                  .sum().unstack(fill_value=0)).astype(float)

        if normalize:
            tot = heat.values.sum()
            if tot > 0:
                heat = heat / tot

            # build annotation table
            annot_vals = (heat*100).round(0)
            # replace very small cells with 0
            annot_vals = annot_vals.where(heat >= label_min_pct, 0)
            annot = annot_vals.astype(int).astype(str) + '%'

            vmin, vmax = 0, 1
            cbar_label = "% of strategy notional"
        else:
            annot = None
            vmin, vmax = 0, vmax_abs
            cbar_label = "Total notional (USD)"

        sns.heatmap(
            heat, ax=ax, cmap="Blues", vmin=vmin, vmax=vmax,
            annot=annot if normalize else False, fmt='',
            annot_kws={'fontsize':10, 'color':'black'},
            linewidths=0.5, linecolor='white',
            cbar_kws={'label': cbar_label}
        )

        ax.set_title(strat, fontsize=13, pad=8)
        ax.set_xlabel("Strike% bucket")
        ax.set_ylabel("Tenor bucket")
        ax.tick_params(axis='x', labelrotation=45, labelsize=9)
        ax.tick_params(axis='y', labelsize=9)

    fig.suptitle("Strike% × Tenor – Notional per Strategy", fontsize=15, y=1.02)
    plt.tight_layout()
    plt.show()
