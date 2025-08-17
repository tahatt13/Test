import pandas as pd
import numpy as np

df = final_data.copy()

# --- Buckets stratégie directement depuis STRATEGY_TYPE
def map_bucket(s: str):
    s = (s or "").lower()
    if "euro" in s and "call" in s and "spread" not in s and "collar" not in s: return "European Call"
    if "amer" in s and "call" in s and "spread" not in s and "collar" not in s: return "American Call"
    if "euro" in s and "put"  in s and "spread" not in s and "collar" not in s: return "European Put"
    if "amer" in s and "put"  in s and "spread" not in s and "collar" not in s: return "American Put"
    if "callspread" in s and "collar" not in s: return "CallSpread"
    if "putspread"  in s and "collar" not in s: return "PutSpread"
    if "callspread" in s and "collar" in s: return "CallSpread Collar"
    if "putspread"  in s and "collar" in s: return "PutSpread Collar"
    return "Other"

df['STRAT_BUCKET'] = df['STRATEGY_TYPE'].astype(str).map(map_bucket)

# Harmonisation légère
if 'SIDE' in df.columns: df['SIDE'] = df['SIDE'].astype(str).str.upper().str.strip()
if 'UNDERLYING_INSTRUMENT_TYPE' in df.columns:
    df['UNDERLYING_INSTRUMENT_TYPE'] = df['UNDERLYING_INSTRUMENT_TYPE'].astype(str).str.title().str.strip()

# --- 1) Landscape global (compat pandas)
landscape_raw = df.groupby('STRAT_BUCKET').agg({
    'NOTIONAL_USD':['count','sum'],
    'VEGA_DOLLAR':'sum',
    'GAMMA_DOLLAR':'sum'
}).reset_index()
landscape_raw.columns = ['STRAT_BUCKET','Trades','Notional_USD','Vega_USD','Gamma_USD']
landscape = landscape_raw.sort_values('Notional_USD', ascending=False)
landscape['Share_%'] = 100*landscape['Notional_USD']/landscape['Notional_USD'].sum()

# --- 2) Mix Buy/Sell
side_pvt = df.pivot_table(values='NOTIONAL_USD', index='STRAT_BUCKET', columns='SIDE', aggfunc='sum', fill_value=0)
buy_series = side_pvt[[c for c in side_pvt.columns if str(c).upper()=='BUY'][0]] if any(str(c).upper()=='BUY' for c in side_pvt.columns) else 0
side_mix = side_pvt.copy()
side_mix['Buy_Share_%'] = 100 * buy_series / side_pvt.sum(axis=1)

# --- 3) Mix Index vs Stock
inst_pvt = df.pivot_table(values='NOTIONAL_USD', index='STRAT_BUCKET', columns='UNDERLYING_INSTRUMENT_TYPE', aggfunc='sum', fill_value=0)
idx_series = inst_pvt[[c for c in inst_pvt.columns if str(c).lower()=='index'][0]] if any(str(c).lower()=='index' for c in inst_pvt.columns) else 0
inst_mix = inst_pvt.copy()
inst_mix['Index_Share_%'] = 100 * idx_series / inst_pvt.sum(axis=1)

# --- 4) Tenor EXACT (utilise tes labels, ordonnés par nombre de mois)
ten_m = (df['TENOR'].astype(str).str.lower().str.replace('m','',regex=False))
ten_m_num = pd.to_numeric(ten_m, errors='coerce')
# ordre croissant sur les valeurs numériques disponibles
order_labels = (pd.DataFrame({'label': df['TENOR'].astype(str), 'm': ten_m_num})
                  .dropna().drop_duplicates().sort_values('m')['label'].str.lower().tolist())
df['TENOR_CAT'] = pd.Categorical(df['TENOR'].astype(str).str.lower(), categories=order_labels, ordered=True)

tenor_by_label = pd.pivot_table(df, values='NOTIONAL_USD',
                                index='STRAT_BUCKET', columns='TENOR_CAT',
                                aggfunc='sum', fill_value=0)
tenor_by_label_pct = tenor_by_label.div(tenor_by_label.sum(axis=1), axis=0).mul(100).round(1)

# --- 5) Top underlyings par stratégie
underlying_top = (df.groupby(['STRAT_BUCKET','UNDERLYING_INSTRUMENT_NAME'])['NOTIONAL_USD']
                    .sum().reset_index()
                    .sort_values(['STRAT_BUCKET','NOTIONAL_USD'], ascending=[True,False]))
