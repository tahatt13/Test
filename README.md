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
    figsize_base=(15, 6),   # <- each subplot will use this size
    show_every_x_label=2
):
    d = df.copy()

    # (same preprocessing as before...)

    # Strategy selection
    if strategies is None:
        strat_order = (d.groupby('STRATEGY_TYPE')['NOTIONAL_USD']
                         .sum().sort_values(ascending=False).index.tolist())
        strategies = strat_order[:top_strategies]

    n_strats = len(strategies)
    fig, axes = plt.subplots(n_strats, 1,
                             figsize=(figsize_base[0], figsize_base[1]*n_strats),
                             squeeze=False)

    for i, strat in enumerate(strategies):
        ax = axes[i, 0]

        # ... build `heat` as before ...

        sns.heatmap(
            heat, ax=ax, cmap="Blues", square=True,
            annot=annot if normalize else False, fmt='',
            annot_kws={'fontsize':12},   # larger annotation font
            linewidths=0.5, linecolor='white',
            cbar_kws={'label': cbar_label}
        )

        ax.set_title(strat, fontsize=16, pad=10)
        ax.set_xlabel("Strike% bucket", fontsize=13)
        ax.set_ylabel("Tenor bucket", fontsize=13)
        ax.tick_params(axis='x', labelrotation=45, labelsize=11)
        ax.tick_params(axis='y', labelsize=11)

    fig.suptitle("Strike% × Tenor – Notional per strategy", fontsize=18, y=0.995)
    plt.tight_layout()
    plt.show()
