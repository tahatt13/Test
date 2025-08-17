import pandas as pd
import numpy as np

df = final_data.copy()

# --- 0) Construire les 7/8 buckets directement à partir de STRATEGY_TYPE
def map_bucket(s: str):
    s = (s or "").lower()
    if "euro" in s and "call" in s and "spread" not in s and "collar" not in s:
        return "European Call"
    if "amer" in s and "call" in s and "spread" not in s and "collar" not in s:
        return "American Call"
    if "euro" in s and "put" in s and "spread" not in s and "collar" not in s:
        return "European Put"
    if "amer" in s and "put" in s and "spread" not in s and "collar" not in s:
        return "American Put"
    if "callspread" in s and "collar" not in s:
        return "CallSpread"
    if "putspread" in s and "collar" not in s:
        return "PutSpread"
    if "callspread" in s and "collar" in s:
        return "CallSpread Collar"
    if "putspread" in s and "collar" in s:
        return "PutSpread Collar"
    return "Other"

df['STRAT_BUCKET'] = df['STRATEGY_TYPE'].astype(str).map(map_bucket)

# --- 1) Landscape global par STRAT_BUCKET
landscape = (df.groupby('STRAT_BUCKET')
               .agg(Trades=('NOTIONAL_USD','count'),
                    Notional_USD=('NOTIONAL_USD','sum'),
                    Vega_USD=('VEGA_DOLLAR','sum'),
                    Gamma_USD=('GAMMA_DOLLAR','sum'))
               .sort_values('Notional_USD', ascending=False))
landscape['Share_%'] = 100*landscape['Notional_USD']/landscape['Notional_USD'].sum()

# --- 2) Mix Buy/Sell
side_mix = (df.groupby(['STRAT_BUCKET','SIDE'])['NOTIONAL_USD']
              .sum().unstack(fill_value=0))
side_mix['Buy_Share_%'] = 100*side_mix.get('BUY',0)/side_mix.sum(axis=1)

# --- 3) Mix Index vs Stock
inst_mix = (df.groupby(['STRAT_BUCKET','UNDERLYING_INSTRUMENT_TYPE'])['NOTIONAL_USD']
              .sum().unstack(fill_value=0))
inst_mix['Index_Share_%'] = 100*inst_mix.get('Index',0)/inst_mix.sum(axis=1)

# --- 4) Répartition tenor (en buckets)
bins_tenor = [0, 1/12, 0.25, 0.5, 1.0, np.inf]
labels_tenor = ['<1m','1-3m','3-6m','6-12m','>12m']
df['TENOR_BUCKET'] = pd.cut(df['TENOR'], bins=bins_tenor, labels=labels_tenor, include_lowest=True)

tenor_pivot_pct = pd.pivot_table(df, values='NOTIONAL_USD',
                                 index='STRAT_BUCKET', columns='TENOR_BUCKET',
                                 aggfunc='sum', fill_value=0)
tenor_pivot_pct = tenor_pivot_pct.div(tenor_pivot_pct.sum(axis=1), axis=0).multiply(100).round(1)

# --- 5) Top underlyings par stratégie
underlying_top = (df.groupby(['STRAT_BUCKET','UNDERLYING_INSTRUMENT_NAME'])['NOTIONAL_USD']
                    .sum().reset_index()
                    .sort_values(['STRAT_BUCKET','NOTIONAL_USD'], ascending=[True,False]))
