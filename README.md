Case	Indicator Conditions	Product Suggestion (Construction)	Exit Condition	Why This Works
Bull – Low IV – Strong Trend	Momentum > 0; Price above SMA50 > SMA200; ADX ≥ 20; IV in bottom 30%	Long Call (Δ≈0.35, DTE 30–60) or Bull Call Debit Spread (Buy Δ≈0.35 call, Sell Δ≈0.25 call, width 5–10% of spot)	Stop if premium loses 50% or trend breaks (price < SMA50). Exit at ≤21 DTE. Take profit at 50–75% of max gain.	Positive delta + long vega benefits from cheap options. Spread limits theta/cost.
Bull – Neutral IV – Strong Trend	Same as above but IV in 30–70% range	Bull Call Debit Spread (Buy Δ≈0.35, Sell Δ≈0.25, width 5–10%, DTE 30–60)	Trend break or 40–50% loss. Exit at ≤21 DTE. TP at 50–75% of max profit.	Good balance of risk/reward when options are fairly priced.
Bull – High IV – Strong Trend	Momentum > 0; Trend bullish; ADX ≥ 20; IV in top 30%	Bull Put Credit Spread (Sell Δ≈0.35 put, Buy Δ≈0.20 put, width 5–10%, DTE 30–45)	Stop if price closes below short strike or loss = 1.5× credit. Exit ≤21 DTE or TP 50–70% of credit.	Keeps bullish delta but profits from selling expensive volatility. Defined downside.
Bull – Weak Trend	Momentum > 0; Trend unclear or ADX < 20	Conservative Bull Call Debit Spread (narrow width 3–6%, DTE 30–45) or No Trade	Exit if momentum flips negative or SMA50 breaks. TP 30–50% of max gain. Exit ≤21 DTE.	When conviction is low, keep exposure small and risk defined.
Bear – Low IV – Strong Trend	Momentum < 0; Price < SMA50 < SMA200; ADX ≥ 20; IV low (≤30%)	Long Put (Δ≈–0.35, DTE 30–60) or Bear Put Debit Spread (Buy Δ≈–0.35, Sell Δ≈–0.25, width 5–10%)	Stop if premium loses 50% or trend breaks (price > SMA50). Exit ≤21 DTE. TP 50–75% of max gain.	Bearish delta + long vega benefits from cheap puts.
Bear – Neutral IV – Strong Trend	Same as above but IV in 30–70%	Bear Put Debit Spread (Δ≈–0.35 vs –0.25, width 5–10%, DTE 30–60)	Trend break or 40–50% loss. Exit ≤21 DTE. TP 50–75%.	Controlled risk exposure with mid-range volatility.
Bear – High IV – Strong Trend	Momentum < 0; Trend bearish; ADX ≥ 20; IV high (>70%)	Bear Call Credit Spread (Sell Δ≈–0.35 call, Buy Δ≈–0.20 call, width 5–10%, DTE 30–45)	Stop if price closes above short strike or loss = 1.5× credit. TP 50–70% credit. Exit ≤21 DTE.	Negative delta + short vega profits from high volatility in downtrends.
Bear – Weak Trend	Momentum < 0; ADX < 20 or unclear trend	Conservative Bear Put Debit Spread (width 3–6%, DTE 30–45) or No Trade	Exit if momentum flips positive or SMA50 breaks up. TP 30–50%. Exit ≤21 DTE.	Reduces risk in choppy/noisy markets.
Breakout Confirmation (Bull or Bear)	Price makes new 20-day high (bull) or low (bear), ADX rising, IV neutral–high	Directional Credit Spread: Bull → Bull Put Spread; Bear → Bear Call Spread (width 5–10%, DTE 30–45)	Exit if breakout fails (price back in range) or TP 60–70% credit. Risk = 1.5× credit.	Credit spreads monetize volatility during strong breakouts.
No Trade (Momentum Off)	ADX < 12 and momentum sign unstable; conflicting signals	Stay flat or switch to mean-reversion book	Wait until ADX ≥ 15–20 and momentum stabilizes	Avoids theta bleed and whipsaws in directionless markets.

<img width="945" height="767" alt="image" src="https://github.com/user-attachments/assets/7739591a-857f-4fb2-960d-5f1479fdee47" />


import pandas as pd

# ==== Helpers ====
def _norm_side(x):
    x = str(x).upper()
    if x in ['B', 'BUY', '1', 'LONG']:
        return 'BUY'
    if x in ['S', 'SELL', '-1', 'SHORT']:
        return 'SELL'
    return x

def _norm_opt(x):
    x = str(x).upper()
    if x.startswith('C'):
        return 'CALL'
    if x.startswith('P'):
        return 'PUT'
    return x

def _to_dt(x):
    return pd.to_datetime(x).normalize()

def classify_spread_type(legs_df: pd.DataFrame) -> str:
    """
    legs_df: DataFrame (2 lignes) pour un quote_id donné, colonnes minimales:
      ['option_type', 'side', 'strike', 'expiry']
    Règles demandées:
      - Call/Put Vertical Bull/Bear (même expiry, strikes ≠)
      - Call/Put Calendar (même strike, expiry ≠)
      - Diagonals (strike ≠ & expiry ≠) -> 'OTHER' (à exclure plus tard)
      - Tout le reste -> 'OTHER'
    """
    df = legs_df.copy()

    # Normalisations
    df['option_type'] = df['option_type'].map(_norm_opt)
    df['side']        = df['side'].map(_norm_side)
    df['expiry']      = df['expiry'].map(_to_dt)
    df['strike']      = pd.to_numeric(df['strike'], errors='coerce')

    # Garde uniquement 2 jambes valides
    df = df.dropna(subset=['option_type','side','strike','expiry'])
    if len(df) != 2 or not set(df['side']).issubset({'BUY','SELL'}):
        return 'OTHER'

    # Décompose
    buy  = df[df['side']=='BUY'].iloc[0]  if (df['side']=='BUY').any()  else None
    sell = df[df['side']=='SELL'].iloc[0] if (df['side']=='SELL').any() else None
    if buy is None or sell is None:
        return 'OTHER'

    opt_buy,  opt_sell  = buy['option_type'],  sell['option_type']
    k_buy,    k_sell    = float(buy['strike']), float(sell['strike'])
    t_buy,    t_sell    = buy['expiry'],        sell['expiry']

    same_strike = (k_buy == k_sell)
    same_expiry = (t_buy == t_sell)

    # ---- Calendars (même strike, expiries ≠) ----
    if same_strike and not same_expiry:
        if opt_buy == 'CALL' and opt_sell == 'CALL':
            return 'CALL CALENDAR SPREAD'
        if opt_buy == 'PUT'  and opt_sell == 'PUT':
            return 'PUT CALENDAR SPREAD'
        # si mix d'option_type (anormal) -> OTHER
        return 'OTHER'

    # ---- Verticals (même expiry, strikes ≠) ----
    if same_expiry and not same_strike:
        # CALL verticals
        if opt_buy == 'CALL' and opt_sell == 'CALL':
            # Bear = BUY high, SELL low ; Bull = SELL high, BUY low
            if k_buy > k_sell:
                return 'CALL SPREAD BEAR'
            elif k_sell > k_buy:
                return 'CALL SPREAD BULL'
            else:
                return 'OTHER'
        # PUT verticals
        if opt_buy == 'PUT' and opt_sell == 'PUT':
            if k_buy > k_sell:
                return 'PUT SPREAD BEAR'
            elif k_sell > k_buy:
                return 'PUT SPREAD BULL'
            else:
                return 'OTHER'
        # mix CALL/PUT au même expiry -> OTHER
        return 'OTHER'

    # ---- Diagonals (strikes ≠ & expiries ≠) -> OTHER (à exclure plus tard) ----
    if (not same_strike) and (not same_expiry):
        return 'OTHER'

    # Cas résiduels
    return 'OTHER'


# ==== Pipeline d’annotation ====
# data_2 contient les jambes (legs) avec au moins: quote_id, option_type, side, strike, expiry
# FINAL_data contient au moins: quote_id

def build_strategy_types(FINAL_data: pd.DataFrame, data_2: pd.DataFrame) -> pd.DataFrame:
    # restreindre aux legs d'options (si tu as aussi des futures/autres)
    legs = data_2.copy()

    # Optionnel: ne garder que CALL/PUT
    legs = legs[legs['option_type'].astype(str).str.upper().str.startswith(('C','P'))]

    # On garde uniquement les quotes qui ont 2 jambes
    leg_counts = legs.groupby('quote_id').size()
    valid_quotes = leg_counts[leg_counts == 2].index

    legs2 = legs[legs['quote_id'].isin(valid_quotes)].copy()

    # Classifier par quote_id
    strategy_rows = []
    for qid, g in legs2.groupby('quote_id'):
        stype = classify_spread_type(g)
        strategy_rows.append({'quote_id': qid, 'STRATEGY_TYPE': stype})

    strat_df = pd.DataFrame(strategy_rows)

    # Merge dans FINAL_data
    out = FINAL_data.merge(strat_df, on='quote_id', how='left')

    # Par défaut, remplir manquants en OTHER
    out['STRATEGY_TYPE'] = out['STRATEGY_TYPE'].fillna('OTHER')

    return out

# ==== Exemple d’utilisation ====
# FINAL_data = build_strategy_types(FINAL_data, data_2)

# ---- Ensuite, pour "enlever" les diagonals (déjà marquées OTHER) ----
# out_no_diagonals = FINAL_data[FINAL_data['STRATEGY_TYPE'] != 'OTHER'].copy()

----
import pandas as pd

def classify_putspread(legs_df):
    """
    legs_df: DataFrame avec les deux jambes d'un put spread
             Colonnes nécessaires: ['strike', 'expiry', 'side'] 
             (side = 'BUY' ou 'SELL')
    """
    if len(legs_df) != 2:
        return "Not a 2-leg spread"
    
    leg1, leg2 = legs_df.iloc[0], legs_df.iloc[1]
    
    # même strike ?
    if leg1['strike'] == leg2['strike']:
        # comparer maturités
        if leg1['expiry'] != leg2['expiry']:
            return "Put Calendar Spread"
    
    # même maturité
    if leg1['expiry'] == leg2['expiry']:
        strikes = sorted([(leg1['strike'], leg1['side']), (leg2['strike'], leg2['side'])])
        # strike bas et haut
        (low_strike, low_side), (high_strike, high_side) = strikes
        
        # bull put spread : sell high strike, buy low strike
        if low_side == 'BUY' and high_side == 'SELL':
            return "Bull Put Spread"
        
        # bear put spread : buy high strike, sell low strike
        if low_side == 'SELL' and high_side == 'BUY':
            return "Bear Put Spread"
    
    return "Unknown / Complex Put Spread"

# Exemple d’utilisation
quote_id = 12345  # un exemple
legs = data_2[data_2['quote_id'] == quote_id]
spread_type = classify_putspread(legs)
print(f"Quote {quote_id}: {spread_type}")

-----
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = final_data.copy()

# --- 0) Harmoniser (léger) les libellés d’underlying
if 'UNDERLYING_INSTRUMENT_TYPE' in df.columns:
    u = df['UNDERLYING_INSTRUMENT_TYPE'].astype(str).str.strip().str.lower()
    # map souple vers Index / Stock / Other
    df['UNDER_T_clean'] = np.select(
        [
            u.str.contains('index'),                      # "Index", "Indice", "Index Option", etc.
            u.str.contains('stock|equity|share|action')   # "Stock", "Equity", etc.
        ],
        ['Index', 'Stock'],
        default='Other'
    )
else:
    raise ValueError("Colonne UNDERLYING_INSTRUMENT_TYPE manquante")

# --- 1) Choisir la colonne de stratégie que tu utilises (décommente la bonne)
STRAT_COL = 'STRATEGY_TYPE'
# STRAT_COL = 'STRAT_BUCKET'   # si tu préfères utiliser le bucket que tu as construit

# --- 2) Tableau Notional par (Stratégie × Type d’underlying)
pivot_abs = pd.pivot_table(
    df, values='NOTIONAL_USD',
    index=STRAT_COL, columns='UNDER_T_clean',
    aggfunc='sum', fill_value=0
)

# S'assurer que les colonnes existent
for col in ['Index','Stock','Other']:
    if col not in pivot_abs.columns:
        pivot_abs[col] = 0.0

# --- 3) % par stratégie (ligne) = notional_col / notional_total_ligne
total = pivot_abs.sum(axis=1)
pct = pivot_abs.div(total, axis=0).mul(100).round(1)

# On ne garde que Index / Stock (mais tu peux afficher Other si utile)
mix_pct = pct[['Index','Stock']].copy()

# --- 4) Ordonner les stratégies par notional total décroissant (plus lisible)
mix_pct = mix_pct.loc[total.sort_values(ascending=False).index]

print("\n=== % de notional par type d’underlying (par stratégie) ===")
print(mix_pct)

# --- 5) (Option) Barres empilées pour visualiser
plt.figure(figsize=(10,6))
bottom = np.zeros(len(mix_pct))
for col in ['Index','Stock']:
    plt.bar(mix_pct.index, mix_pct[col].values, bottom=bottom, label=col)
    bottom += mix_pct[col].values
plt.ylabel('% du notional')
plt.title('Répartition Index vs Stock par stratégie')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.show()

here -----
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
