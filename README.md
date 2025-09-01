import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def strategy_heatmaps(
    df,
    cluster_id=None,
    strategies=None,
    strike_bins=None,                 # liste de bornes numériques
    strike_labels=None,               # libellés str (optionnel)
    tenor_bins_bd=[0,21,63,126,252,504,756],
    tenor_labels=['<1M','1–3M','3–6M','6–12M','1–2Y','2–3Y'],
    normalize=True,
    min_cell_value=0.01
):
    d = df.copy()

    # filtre éventuel sur un cluster
    if cluster_id is not None and 'cluster_kmeans' in d.columns:
        d = d[d['cluster_kmeans'] == cluster_id].copy()

    # numeric safe
    num_cols = ['NOTIONAL_USD','STRIKE_PCT','STRIKE_2','MATURITY_in_BDAYS','MATURITY_WIDTH','NUMBER_OF_LEGS']
    for c in num_cols:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors='coerce')

    # midpoints (single vs multi)
    is_single = (d['NUMBER_OF_LEGS'] == 1)
    d['_strike_mid'] = np.where(is_single, d['STRIKE_PCT'],
                                (d['STRIKE_PCT'] + d['STRIKE_2'])/2.0)
    d['_mat_mid_bd'] = np.where(is_single, d['MATURITY_in_BDAYS'],
                                d['MATURITY_in_BDAYS'] + d['MATURITY_WIDTH']/2.0)

    d = d.dropna(subset=['_strike_mid','_mat_mid_bd','NOTIONAL_USD','STRATEGY_TYPE'])

    # --- Buckets strike & tenor (avec catégories ordonnées) ---
    if strike_bins is None:
        strike_bins = np.arange(60, 141, 5)
    d['_strike_bucket'] = pd.cut(d['_strike_mid'], bins=strike_bins, include_lowest=True,
                                 labels=strike_labels if strike_labels is not None else None, right=False)
    d['_tenor_bucket']  = pd.cut(d['_mat_mid_bd'], bins=tenor_bins_bd, include_lowest=True,
                                 labels=tenor_labels, right=False)

    # s'assurer que ce sont des catégories ordonnées
    d['_strike_bucket'] = d['_strike_bucket'].astype('category')
    d['_tenor_bucket']  = d['_tenor_bucket'].astype('category')

    # quelles stratégies
    all_strats = d['STRATEGY_TYPE'].dropna().unique().tolist()
    if strategies is None:
        strategies = all_strats
    else:
        strategies = [s for s in strategies if s in all_strats]
    if not strategies:
        print("No matching STRATEGY_TYPE to plot."); return

    n = len(strategies)
    cols = 3
    rows = int(np.ceil(n/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 3.8*rows), squeeze=False)

    # global max si non normalisé (même échelle couleur)
    vmax_global = None
    if not normalize:
        tmp = (d.groupby(['STRATEGY_TYPE','_tenor_bucket','_strike_bucket'])['NOTIONAL_USD']
                 .sum().reset_index())
        vmax_global = float(tmp['NOTIONAL_USD'].max())

    for i, strat in enumerate(strategies):
        ax = axes[i//cols, i%cols]
        dd = d[d['STRATEGY_TYPE'] == strat]

        if dd.empty:
            ax.axis('off'); ax.set_title(f"{strat} (no data)"); continue

        heat = (dd.groupby(['_tenor_bucket','_strike_bucket'])['NOTIONAL_USD']
                  .sum()
                  .unstack(fill_value=0))

        # >>> IMPORTANT : caster en float avant toute opération
        heat = heat.astype('float64')

        if normalize:
            denom = heat.values.sum()
            if denom > 0:
                heat = heat / denom
            if min_cell_value > 0:
                heat = heat.mask(heat < min_cell_value)
            cbar_label = '% of strategy notional'
            vmin, vmax = 0, 1
        else:
            if min_cell_value > 0:
                heat = heat.mask(heat < min_cell_value)
            cbar_label = 'Total notional (USD)'
            vmin, vmax = 0, vmax_global

        sns.heatmap(heat, ax=ax, cmap='Blues', vmin=vmin, vmax=vmax,
                    cbar_kws={'label': cbar_label})
        ax.set_title(strat)
        ax.set_xlabel('Strike% bucket')
        ax.set_ylabel('Tenor bucket')

    # masquer sous-graphiques vides
    for j in range(i+1, rows*cols):
        axes[j//cols, j%cols].axis('off')

    fig.suptitle("Strike% × Tenor – Notional per strategy", fontsize=14)
    plt.tight_layout(rect=[0,0,1,0.96])
    plt.show()


bins_strike = [0,10,20,30,40,50,60,70,80,90,95,96,97,98,99,100,
               101,102,103,104,105,110,120,np.inf]
labels_strike = ['0-10%','10-20%','20-30%','30-40%','40-50%','50-60%',
                 '60-70%','70-80%','80-90%','90-95%','95-96%','96-97%',
                 '97-98%','98-99%','99-100%','100-101%','101-102%','102-103%',
                 '103-104%','104-105%','105-110%','110-120%','120%+']

strategy_heatmaps(
    df,
    cluster_id=16,
    strategies=['European Call','European Put','Bull Call Spread','Bear Put Spread'],
    strike_bins=bins_strike,
    strike_labels=labels_strike,
    normalize=True,
    min_cell_value=0.01
)
