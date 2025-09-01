import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def strategy_heatmaps_one_per_line(
    df,
    cluster_id=None,
    strategies=None,              # list of str, or None for top N
    top_strategies=6,             # if strategies=None
    strike_bins=None,
    strike_labels=None,
    tenor_bins_bd=[0,21,63,126,252,504,756],
    tenor_labels=['<1M','1–3M','3–6M','6–12M','1–2Y','2–3Y'],
    normalize=True,
    label_min_pct=0.02,           # show labels ≥ 2% only
    figsize_per=(10,4)            # width, height per heatmap
):
    d = df.copy()

    # filter on cluster
    if cluster_id is not None and 'cluster_kmeans' in d.columns:
        d = d[d['cluster_kmeans'] == cluster_id].copy()

    # numeric safe
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

    # buckets
    if strike_bins is None:
        strike_bins = np.arange(90, 111, 2)  # tighter range around ATM
    d['_strike_bucket'] = pd.cut(d['_strike_mid'], bins=strike_bins, include_lowest=True,
                                 labels=strike_labels if strike_labels else None, right=False)
    d['_tenor_bucket']  = pd.cut(d['_mat_mid_bd'], bins=tenor_bins_bd, include_lowest=True,
                                 labels=tenor_labels, right=False)

    # strategy selection
    if strategies is None:
        strat_order = (d.groupby('STRATEGY_TYPE')['NOTIONAL_USD']
                         .sum().sort_values(ascending=False).index.tolist())
        strategies = strat_order[:top_strategies]
    else:
        have = set(d['STRATEGY_TYPE'].unique())
        strategies = [s for s in strategies if s in have]

    if not strategies:
        print("No matching STRATEGY_TYPE to plot."); return

    # global vmax if not normalized
    vmax_abs = 0.0
    if not normalize:
        tmp = (d.groupby(['STRATEGY_TYPE','_tenor_bucket','_strike_bucket'])['NOTIONAL_USD']
                 .sum().reset_index())
        vmax_abs = float(tmp['NOTIONAL_USD'].max())

    # plot each strategy one per row
    fig, axes = plt.subplots(len(strategies), 1,
                             figsize=(figsize_per[0], figsize_per[1]*len(strategies)),
                             squeeze=False)

    for i, strat in enumerate(strategies):
        ax = axes[i,0]
        dd = d[d['STRATEGY_TYPE'] == strat]

        heat = (dd.groupby(['_tenor_bucket','_strike_bucket'])['NOTIONAL_USD']
                  .sum().unstack(fill_value=0)).astype(float)

        annot = None
        if normalize:
            tot = heat.values.sum()
            if tot > 0:
                heat = heat / tot
            # labels only for cells >= threshold
            annot = (heat*100).round(0).astype('Int64').astype(str) + '%'
            annot = annot.where(heat >= label_min_pct, '')
            vmin, vmax = 0, 1
            cbar_label = "% of strategy notional"
        else:
            vmin, vmax = 0, vmax_abs
            cbar_label = "Total notional (USD)"

        sns.heatmap(
            heat, ax=ax, cmap="Blues", vmin=vmin, vmax=vmax,
            annot=annot if normalize else False, fmt='',
            annot_kws={'fontsize':10},
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
