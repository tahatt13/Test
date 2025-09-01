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
