import pandas as pd
import numpy as np

df = final_data.copy()

# --- 0) Buckets à partir de STRATEGY_TYPE
def map_bucket(s: str):
    s = (s or "").lower()
    if "euro" in s and "call" in s and "spread" not in s and "collar" not in s:
        return "European Call"
    if "amer" in s and "call" in s and "spread" not in s and "collar" not in s:
        return "American Call"
    if "euro" in s and "put"  in s and "spread" not in s and "collar" not in s:
        return "European Put"
    if "amer" in s and "put"  in s and "spread" not in s and "collar" not in s:
        return "American Put"
    if "callspread" in s and "collar" not in s:
        return "CallSpread"
    if "putspread"  in s and "collar" not in s:
        return "PutSpread"
    if "callspread" in s and "collar" in s:
        return "CallSpread Collar"
    if "putspread"  in s and "collar" in s:
        return "PutSpread Collar"
    return "Other"

df['STRAT_BUCKET'] = df['STRATEGY_TYPE'].astype(str).map(map_bucket)

# Harmonise SIDE & UNDERLYING_INSTRUMENT_TYPE
if 'SIDE' in df.columns:
    df['SIDE'] = df['SIDE'].astype(str).str.upper().str.strip()
if 'UNDERLYING_INSTRUMENT_TYPE' in df.columns:
    df['UNDERLYING_INSTRUMENT_TYPE'] = df['UNDERLYING_INSTRUMENT_TYPE'].astype(str).str.title().str.strip()

# --- 1) Landscape global (compat dict-agg)
landscape_raw = df.groupby('STRAT_BUCKET').agg({
    'NOTIONAL_USD': ['count','sum'],
    'VEGA_DOLLAR': 'sum',
    'GAMMA_DOLLAR': 'sum'
}).reset_index()

# renomme proprement
landscape_raw.columns = ['STRAT_BUCKET','Trades','Notional_USD','Vega_USD','Gamma_USD']
landscape = landscape_raw.sort_values('Notional_USD', ascending=False)
landscape['Share_%'] = 100 * landscape['Notional_USD'] / landscape['Notional_USD'].sum()

# --- 2) Mix Buy/Sell
side_pvt = df.pivot_table(values='NOTIONAL_USD',
                          index='STRAT_BUCKET',
                          columns='SIDE',
                          aggfunc='sum',
                          fill_value=0)
# robust au cas où 'BUY' n'existe pas
buy_col = [c for c in side_pvt.columns if str(c).upper()=='BUY']
buy_series = side_pvt[buy_col[0]] if buy_col else 0
side_mix = side_pvt.copy()
side_mix['Buy_Share_%'] = 100 * buy_series / side_pvt.sum(axis=1)

# --- 3) Mix Index vs Stock
inst_pvt = df.pivot_table(values='NOTIONAL_USD',
                          index='STRAT_BUCKET',
                          columns='UNDERLYING_INSTRUMENT_TYPE',
                          aggfunc='sum',
                          fill_value=0)
idx_col = [c for c in inst_pvt.columns if str(c).lower()=='index']
idx_series = inst_pvt[idx_col[0]] if idx_col else 0
inst_mix = inst_pvt.copy()
inst_mix['Index_Share_%'] = 100 * idx_series / inst_pvt.sum(axis=1)

# --- 4) Tenor buckets
bins_tenor = [0, 1/12.0, 0.25, 0.5, 1.0, np.inf]
labels_tenor = ['<1m','1-3m','3-6m','6-12m','>12m']
df['TENOR_BUCKET'] = pd.cut(df['TENOR'], bins=bins_tenor, labels=labels_tenor, include_lowest=True)

tenor_pivot = pd.pivot_table(df, values='NOTIONAL_USD',
                             index='STRAT_BUCKET', columns='TENOR_BUCKET',
                             aggfunc='sum', fill_value=0)
tenor_pivot_pct = tenor_pivot.div(tenor_pivot.sum(axis=1), axis=0).multiply(100).round(1)

# --- 5) Top underlyings par stratégie
underlying_top = (df.groupby(['STRAT_BUCKET','UNDERLYING_INSTRUMENT_NAME'])['NOTIONAL_USD']
                    .sum()
                    .reset_index()
                    .sort_values(['STRAT_BUCKET','NOTIONAL_USD'], ascending=[True,False]))
