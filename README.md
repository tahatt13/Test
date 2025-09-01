for strat in df_single_straddle["STRATEGY_TYPE"].unique():
    dd = df_single_straddle[df_single_straddle["STRATEGY_TYPE"] == strat]

    # pivot table of notional
    heat = (dd.groupby(["_tenor_bucket","_strike_bucket"])["NOTIONAL_USD"]
              .sum()
              .unstack(fill_value=0))

    # ✅ force column order = labels
    heat = heat.reindex(columns=labels, fill_value=0)

    # normalize
    heat = heat / heat.values.sum()

    # annotations
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
