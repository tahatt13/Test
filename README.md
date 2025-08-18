
import numpy as np

df = df.copy()

# helpers pour percentiles (nommés pour éviter <lambda>)
def p95(x): return np.percentile(x, 95)
def p5(x):  return np.percentile(x, 5)

g = df.groupby('STRAT_BUCKET')

summary = g.agg({
    'NOTIONAL_USD': ['count', 'mean', p95],
    'TENOR_M': ['mean'],
    'STRIKE_PCT_CLEAN': ['mean', p5, p95],
    'VEGA_DOLLAR': ['sum'],
    'GAMMA_DOLLAR': ['sum'],
})

# aplatir les colonnes MultiIndex -> noms clairs
summary.columns = [
    f"{c0}_{(c1 if isinstance(c1,str) else c1.__name__)}"
    for c0, c1 in summary.columns
]

# renommer pour matcher exactement ce qu’on voulait
summary = summary.rename(columns={
    'NOTIONAL_USD_count': 'n_trades',
    'NOTIONAL_USD_mean':  'notional_mean',
    'NOTIONAL_USD_p95':   'notional_p95',
    'TENOR_M_mean':       'tenor_mean',
    'STRIKE_PCT_CLEAN_mean': 'strike_pct_mean',
    'STRIKE_PCT_CLEAN_p5':   'strike_pct_p5',
    'STRIKE_PCT_CLEAN_p95':  'strike_pct_p95',
    'VEGA_DOLLAR_sum':    'vega_sum',
    'GAMMA_DOLLAR_sum':   'gamma_sum',
}).round(2)

print(summary)

----------
f = final_data.copy()

# 1) Ta version calculée (strike / spot_px)
df['STRIKE_OVER_SPOTPX'] = df['STRIKE'] / df['SPOT_PX']

# 2) Ta colonne déjà existante (en % → on repasse en ratio)
df['STRIKE_PCT_EXISTING'] = df['STRIKE_PCT_SPOT'] / 100

# 3) Choisir la valeur "sûre" : on prend la plus petite des deux
df['STRIKE_PCT_CLEAN'] = np.minimum(df['STRIKE_OVER_SPOTPX'], df['STRIKE_PCT_EXISTING'])

# -------- 1) Distribution STRIKE%SPOT par stratégie --------
plt.figure(figsize=(10,6))
sns.kdeplot(data=df, x='STRIKE_PCT_CLEAN', hue='STRAT_BUCKET', common_norm=False)
plt.title("Distribution du Strike/Spot par stratégie (corrigé)")
plt.xlabel("Strike / Spot (ratio)")
plt.show()

# -------- 3) Scatter STRIKE%SPOT vs TENOR --------
plt.figure(figsize=(10,6))
sns.scatterplot(data=df.sample(min(5000, len(df)), random_state=1), 
                x='STRIKE_PCT_CLEAN', y='TENOR_M', 
                hue='STRAT_BUCKET', alpha=0.4)
plt.title("Strike/Spot vs Tenor (corrigé)")
plt.xlabel("Strike / Spot")
plt.ylabel("Maturité (mois)")
plt.show()

# -------- 4) Stats descriptives --------
summary = df.groupby('STRAT_BUCKET').agg(
    n_trades=('NOTIONAL_USD','count'),
    notional_mean=('NOTIONAL_USD','mean'),
    notional_p95=('NOTIONAL_USD', lambda x: np.percentile(x,95)),
    tenor_mean=('TENOR_M','mean'),
    strike_pct_mean=('STRIKE_PCT_CLEAN','mean'),
    strike_pct_p5=('STRIKE_PCT_CLEAN', lambda x: np.percentile(x,5)),
    strike_pct_p95=('STRIKE_PCT_CLEAN', lambda x: np.percentile(x,95)),
    vega_sum=('VEGA_DOLLAR','sum'),
    gamma_sum=('GAMMA_DOLLAR','sum'),
).round(2)

print(summary)
