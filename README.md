import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = final_data.copy()

# ======================
# 1) Profil par client
# ======================
client_profile = df.groupby('CLIENT_NAME').agg({
    'NOTIONAL_USD':'sum',
    'VEGA_DOLLAR':'sum',
    'GAMMA_DOLLAR':'sum',
    'STRAT_BUCKET':'nunique',
    'UNDERLYING_INSTRUMENT_TYPE':'nunique'
}).rename(columns={
    'NOTIONAL_USD':'Total_Notional',
    'VEGA_DOLLAR':'Total_Vega',
    'GAMMA_DOLLAR':'Total_Gamma',
    'STRAT_BUCKET':'Nb_Strategies',
    'UNDERLYING_INSTRUMENT_TYPE':'Nb_Underlying_Types'
}).sort_values('Total_Notional', ascending=False)

print("\n=== Top 15 Clients (profil global) ===")
print(client_profile.head(15))

# ======================
# 2) Mix stratégie par client
# ======================
client_strat = pd.pivot_table(df, values='NOTIONAL_USD',
                              index='CLIENT_NAME', columns='STRAT_BUCKET',
                              aggfunc='sum', fill_value=0)
client_strat_pct = client_strat.div(client_strat.sum(axis=1), axis=0).round(3)

print("\n=== Mix Stratégies – Top 10 Clients ===")
print(client_strat_pct.head(10))

# ======================
# 3) Profil par type de sous-jacent
# ======================
under_type = df.groupby('UNDERLYING_INSTRUMENT_TYPE')['NOTIONAL_USD'].sum()

under_strat = pd.pivot_table(df, values='NOTIONAL_USD',
                             index='UNDERLYING_INSTRUMENT_TYPE', columns='STRAT_BUCKET',
                             aggfunc='sum', fill_value=0)
under_strat_pct = under_strat.div(under_strat.sum(axis=1), axis=0).mul(100).round(1)

print("\n=== Répartition Notional par type d’underlying ===")
print(under_type)

print("\n=== Mix Stratégies par type d’underlying (en %) ===")
print(under_strat_pct)

# ======================
# 4) Top underlyings détaillés
# ======================
top_underlyings = df.groupby('UNDERLYING_INSTRUMENT_NAME')['NOTIONAL_USD'] \
                    .sum().sort_values(ascending=False).head(20)

print("\n=== Top 20 Underlyings ===")
print(top_underlyings)

# ======================
# 5) Heatmap Client × Stratégie (optionnel visuel)
# ======================
top_clients = client_profile.head(10).index
pivot = pd.pivot_table(df[df['CLIENT_NAME'].isin(top_clients)],
                       values='NOTIONAL_USD',
                       index='CLIENT_NAME', columns='STRAT_BUCKET',
                       aggfunc='sum', fill_value=0)
pivot_pct = pivot.div(pivot.sum(axis=1), axis=0)

plt.figure(figsize=(10,6))
sns.heatmap(pivot_pct, annot=True, cmap="Blues", fmt=".0%")
plt.title("Top 10 Clients – Répartition du notional par stratégie")
plt.show()

----
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
