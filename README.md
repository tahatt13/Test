import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---- pick the data scope ----
# cluster_16 = df[df['cluster_kmeans'] == 16].copy()
data = cluster_16.copy()

# --- standardize the FTSE name ---
def fix_name(u):
    if not isinstance(u, str): return u
    u_up = u.upper()
    if 'FTSE' in u_up and '100' in u_up:
        return 'FTSE100'
    return u.strip()

data['UNDERLYING_INSTRUMENT_NAME'] = data['UNDERLYING_INSTRUMENT_NAME'].map(fix_name)

# --- aggregate notional by strategy x underlying ---
g = (data.groupby(['STRATEGY_TYPE','UNDERLYING_INSTRUMENT_NAME'])['NOTIONAL_USD']
          .sum().reset_index())

# totals (for order and shares)
totals = g.groupby('STRATEGY_TYPE')['NOTIONAL_USD'].sum().rename('TOTAL')
g = g.merge(totals, on='STRATEGY_TYPE')
g['share'] = g['NOTIONAL_USD'] / g['TOTAL']

# keep top-3, add "Other" = remainder
g_sorted = g.sort_values(['STRATEGY_TYPE','NOTIONAL_USD'], ascending=[True, False])
top3 = g_sorted.groupby('STRATEGY_TYPE').head(3)

other = (g_sorted.groupby('STRATEGY_TYPE')['share']
         .apply(lambda s: max(0.0, 1.0 - s.nlargest(3).sum()))
         .reset_index())
other['UNDERLYING_INSTRUMENT_NAME'] = 'Other'

plotdf = pd.concat([
    top3[['STRATEGY_TYPE','UNDERLYING_INSTRUMENT_NAME','share']],
    other[['STRATEGY_TYPE','UNDERLYING_INSTRUMENT_NAME','share']]
], ignore_index=True)

# order strategies by total notional (descending)
order = totals.sort_values(ascending=False).index.tolist()
plotdf['STRATEGY_TYPE'] = pd.Categorical(plotdf['STRATEGY_TYPE'], order)

# pivot to 100% stacked layout
wide = (plotdf.pivot(index='STRATEGY_TYPE',
                     columns='UNDERLYING_INSTRUMENT_NAME',
                     values='share')
        .fillna(0.0).sort_index())

# --- color palette: one color per underlying, "Other" = grey ---
underlyings = list(wide.columns)
palette = sns.color_palette('tab20', n_colors=max(3, len(underlyings)))
color_map = {u: palette[i % len(palette)] for i, u in enumerate(underlyings)}
color_map['Other'] = (0.7, 0.7, 0.7)  # grey for Other

# plot 100% stacked horizontal bars
fig, ax = plt.subplots(figsize=(12, 7))
left = np.zeros(len(wide))

# put "Other" last
cols_order = [c for c in underlyings if c != 'Other'] + (['Other'] if 'Other' in underlyings else [])

for col in cols_order:
    vals = wide[col].values
    ax.barh(wide.index.astype(str), vals, left=left, color=color_map[col], label=col)
    left += vals

# annotate only segments >= 50%
for yi, strat in enumerate(wide.index):
    cum = 0.0
    for col in cols_order:
        val = float(wide.loc[strat, col])
        if val >= 0.50:
            ax.text(cum + val/2, yi, f"{col} • {val:.0%}",
                    ha='center', va='center', fontsize=10, color='white', fontweight='bold')
        cum += val

ax.set_title("Top Underlyings per Strategy — Share of Strategy Notional (100% each)", fontsize=14)
ax.set_xlabel("Share of strategy notional")
ax.set_xlim(0, 1)
ax.legend(title="Underlying", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()
