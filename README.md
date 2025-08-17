import pandas as pd
import numpy as np

df = final_data.copy()

# ---- 0) Normalisation légère des libellés stratégie (sans cleaning)
def parse_strategy(s: str):
    s = (s or "").lower()
    style = None
    family = None

    if "collar" in s and "call" in s:
        family = "CallSpread Collar"
    elif "collar" in s and "put" in s:
        family = "PutSpread Collar"
    elif "spread" in s and "call" in s:
        family = "CallSpread"
    elif "spread" in s and "put" in s:
        family = "PutSpread"
    elif "call" in s:
        family = "Call"
    elif "put" in s:
        family = "Put"

    if "euro" in s:
        style = "European"
    elif "amer" in s:
        style = "American"

    return pd.Series([style, family])

df[['EXERCISE_STYLE','STRAT_FAMILY']] = df['STRATEGY_TYPE'].astype('string').map(parse_strategy)

# Bucket final (pour singles: style+famille ; pour spreads/collars: famille seule)
def make_bucket(row):
    fam = row['STRAT_FAMILY']
    sty = row['EXERCISE_STYLE']
    if fam in ["Call","Put"] and sty is not None:
        return f"{sty} {fam}"
    return fam

df['STRAT_BUCKET'] = df.apply(make_bucket, axis=1)

# ---- 1) Landscape global par STRAT_BUCKET
agg_cols = {
    'NOTIONAL_USD':'sum',
    'VEGA_DOLLAR':'sum',
    'GAMMA_DOLLAR':'sum',
    'QUANTITY':'count'
}
landscape = (df.groupby('STRAT_BUCKET')
               .agg(Trades=('QUANTITY','count'),
                    Notional_USD=('NOTIONAL_USD','sum'),
                    Vega_USD=('VEGA_DOLLAR','sum'),
                    Gamma_USD=('GAMMA_DOLLAR','sum'))
               .sort_values('Notional_USD', ascending=False))
landscape['Share_%'] = 100*landscape['Notional_USD']/landscape['Notional_USD'].sum()

# ---- 2) Mix par SIDE (Buy/Sell) et par instrument type
side_mix = (df.groupby(['STRAT_BUCKET','SIDE'])['NOTIONAL_USD']
              .sum().unstack(fill_value=0))
side_mix['Buy_Share_%'] = 100*side_mix.get('BUY',0)/side_mix.sum(axis=1)

inst_mix = (df.groupby(['STRAT_BUCKET','UNDERLYING_INSTRUMENT_TYPE'])['NOTIONAL_USD']
              .sum().unstack(fill_value=0))
inst_mix['Index_Share_%'] = 100*inst_mix.get('Index',0)/inst_mix.sum(axis=1)

# ---- 3) Tenor buckets (en années) et distribution par STRAT_BUCKET
# (si TENOR est déjà en années, parfait ; sinon adapte)
bins_tenor = [0, 1/12, 0.25, 0.5, 1.0, np.inf]
labels_tenor = ['<1m','1-3m','3-6m','6-12m','>12m']
df['TENOR_BUCKET'] = pd.cut(df['TENOR'], bins=bins_tenor, labels=labels_tenor, include_lowest=True)

tenor_pivot = pd.pivot_table(df, values='NOTIONAL_USD',
                             index='STRAT_BUCKET', columns='TENOR_BUCKET',
                             aggfunc='sum', fill_value=0)
tenor_pivot_pct = tenor_pivot.div(tenor_pivot.sum(axis=1), axis=0).multiply(100).round(1)

# ---- 4) Moneyness buckets (utilise ta colonne déjà binned)
mny_col = 'STRIKE_PCT_SPOT_BIN' if 'STRIKE_PCT_SPOT_BIN' in df.columns else None
if mny_col:
    mny_pivot = pd.pivot_table(df, values='NOTIONAL_USD',
                               index='STRAT_BUCKET', columns=mny_col,
                               aggfunc='sum', fill_value=0)
    mny_pivot_pct = mny_pivot.div(mny_pivot.sum(axis=1), axis=0).multiply(100).round(1)
else:
    mny_pivot = mny_pivot_pct = None

# ---- 5) Style split uniquement pour les singles (European/ American)
style_split = (df[df['STRAT_FAMILY'].isin(['Call','Put'])]
               .groupby(['EXERCISE_STYLE','STRAT_FAMILY'])['NOTIONAL_USD']
               .sum().unstack(fill_value=0))
