import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def heatmaps_single_and_straddle(
    df,
    cluster_id=None,
    strike_bins=np.arange(80, 121, 2),   # 80%–120% in 2% steps
    tenor_bins_bd=[0,21,63,126,252,504,756],
    tenor_labels=['<1M','1–3M','3–6M','6–12M','1–2Y','2–3Y'],
    figsize=(12, 5)
):
    d = df.copy()

    # optional filter
    if cluster_id is not None and 'cluster_kmeans' in d.columns:
        d = d[d['cluster_kmeans'] == cluster_id]

    # keep only single leg or straddle
    mask = ((d['NUMBER_OF_LEGS'] == 1) | (d['STRATEGY_TYPE'] == 'Straddle'))
    d = d[mask].copy()

    # bin strike% and tenor
    d['_strike_bucket'] = pd.cut(d['STRIKE_PCT'], bins=strike_bins, right=False)
    d['_tenor_bucket']  = pd.cut(d['MATURITY_in_BDAYS'],
                                 bins=tenor_bins_bd, labels=tenor_labels, right=False)

    # group notional
    grouped = (d.groupby(['STRATEGY_TYPE','_tenor_bucket','_strike_bucket'])['NOTIONAL_USD']
                 .sum().reset_index())

    # normalize within each strategy to percentages
    grouped['pct'] = grouped.groupby('STRATEGY_TYPE')['NOTIONAL_USD'].transform(lambda x: x / x.sum())

    # pivot for heatmap
    strategies = grouped['STRATEGY_TYPE'].unique()
    n = len(strategies)
    fig, axes = plt.subplots(n, 1, figsize=(figsize[0], figsize[1]*n), squeeze=False)

    for i, strat in enumerate(strategies):
        ax = axes[i,0]
        dd = grouped[grouped['STRATEGY_TYPE'] == strat]
        heat = dd.pivot(index='_tenor_bucket', columns='_strike_bucket', values='pct').fillna(0)

        # annot with %
        annot = (heat*100).round(0).astype(int).astype(str) + '%'

        sns.heatmap(
            heat, ax=ax, cmap="Blues", vmin=0, vmax=1,
            annot=annot, fmt='',
            annot_kws={'fontsize':9, 'color':'black'},
            cbar_kws={'label': "% of strategy notional"}
        )
        ax.set_title(strat, fontsize=13, pad=8)
        ax.set_xlabel("Strike% bucket")
        ax.set_ylabel("Tenor bucket")
        ax.tick_params(axis='x', labelrotation=45)

    fig.suptitle("Strike% × Tenor — Single Legs & Straddle", fontsize=15, y=0.995)
    plt.tight_layout()
    plt.show()
