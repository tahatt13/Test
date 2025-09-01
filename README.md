import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# filter only single legs and straddles
df_single_straddle = df[(df["NUMBER_OF_LEGS"] == 1) | (df["STRATEGY_TYPE"] == "Straddle")].copy()

# define bins
strike_bins = np.arange(80, 121, 2)  # 80%–120% in 2% steps
tenor_bins_bd = [0,21,63,126,252,504,756]
tenor_labels = ['<1M','1–3M','3–6M','6–12M','1–2Y','2–3Y']

# binning
df_single_straddle["_strike_bucket"] = pd.cut(df_single_straddle["STRIKE_PCT"], bins=strike_bins, right=False)
df_single_straddle["_tenor_bucket"]  = pd.cut(df_single_straddle["MATURITY_in_BDAYS"],
                                              bins=tenor_bins_bd, labels=tenor_labels, right=False)

# loop over strategies
for strat in df_single_straddle["STRATEGY_TYPE"].unique():
    dd = df_single_straddle[df_single_straddle["STRATEGY_TYPE"] == strat]

    # aggregate notional and normalize to %
    heat = (dd.groupby(["_tenor_bucket","_strike_bucket"])["NOTIONAL_USD"]
              .sum()
              .unstack(fill_value=0))
    heat = heat / heat.values.sum()  # normalize
    annot = (heat*100).round(0).astype(int).astype(str) + "%"

    # plot
    plt.figure(figsize=(14,6))
    sns.heatmap(heat, cmap="Blues", vmin=0, vmax=1,
                annot=annot, fmt="", annot_kws={"fontsize":10, "color":"black"},
                cbar_kws={"label":"% of strategy notional"})
    plt.title(f"Strike% × Tenor – {strat}", fontsize=16, pad=10)
    plt.xlabel("Strike% bucket")
    plt.ylabel("Tenor bucket")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
