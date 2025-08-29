
import pandas as pd

def compute_trade_pnls(option_df, trades, fv_col="FV", forbid_overlap=True):
    """
    Compute trade-level PnLs and daily MTM equity from option time series.

    Parameters
    ----------
    option_df : DataFrame
        Must contain ['startDate','endDate','date', fv_col].
        One row per day between startDate..endDate inclusive, one contract per startDate.
    trades : list of (entry, exit)
        Output of get_trades_from_signal_with_expiry (expiry-capped).
    fv_col : str, default 'FV'
        Column with the option fair value.
    forbid_overlap : bool, default True
        Skip a trade if its entry <= last trade's exit (no overlapping positions).

    Returns
    -------
    trades_df : DataFrame
        One row per trade with columns:
        [entry, exit, expiry, entry_FV, exit_FV, PnL]
    pnl_ts : DataFrame
        Daily MTM time series across all trades (index=date, col='MTM').
    """

    df = option_df.copy()
    df["startDate"] = pd.to_datetime(df["startDate"])
    df["endDate"]   = pd.to_datetime(df["endDate"])
    df["date"]      = pd.to_datetime(df["date"])

    trades_rows, pnl_chunks = [], []
    last_exit = None

    for entry, exit in trades:
        # strict alignment: get the contract starting at entry
        sub = df[df["startDate"] == entry]
        if sub.empty:
            continue

        expiry = sub["endDate"].iloc[0]
        if exit > expiry:
            exit = expiry  # safeguard, though trades should already be capped

        if forbid_overlap and last_exit is not None and entry <= last_exit:
            continue

        trade_path = sub[(sub["date"] >= entry) & (sub["date"] <= exit)].sort_values("date")
        if trade_path.empty:
            continue

        entry_fv = float(trade_path.loc[trade_path["date"] == entry, fv_col].iloc[0])
        exit_fv  = float(trade_path.loc[trade_path["date"] == exit,   fv_col].iloc[0])
        pnl      = exit_fv - entry_fv

        # Daily MTM vs entry
        tp = trade_path.copy()
        tp["MTM"] = tp[fv_col] - entry_fv
        pnl_chunks.append(tp[["date","MTM"]])

        trades_rows.append({
            "entry": entry,
            "exit": exit,
            "expiry": expiry,
            "entry_FV": entry_fv,
            "exit_FV": exit_fv,
            "PnL": pnl
        })
        last_exit = exit

    trades_df = pd.DataFrame(trades_rows).sort_values("entry") if trades_rows else pd.DataFrame(
        columns=["entry","exit","expiry","entry_FV","exit_FV","PnL"]
    )
    pnl_ts = (
        pd.concat(pnl_chunks).set_index("date").sort_index()
        if pnl_chunks else pd.DataFrame(columns=["MTM"])
    )

    return trades_df, pnl_ts

----
import pandas as pd

def make_expiry_map(entry_exit_dates):
    """
    entry_exit_dates: list of (start, expiry) for ALL possible starts in your backtest
    Returns: dict {startDate: expiryDate}
    """
    df = pd.DataFrame(entry_exit_dates, columns=["startDate","expiry"])
    df["startDate"] = pd.to_datetime(df["startDate"])
    df["expiry"]    = pd.to_datetime(df["expiry"])
    # if duplicates, keep first (or change keep="last")
    df = df.drop_duplicates(subset=["startDate"], keep="first").sort_values("startDate")
    return df.set_index("startDate")["expiry"].to_dict()

def adjust_trades_with_expiry_map(trades, expiry_map):
    """
    trades: list of (entry, exit) from your signal (already shifted)
    expiry_map: dict {startDate: expiryDate}
    Returns: list of (entry, exit_capped, expiry) keeping only entries that exist in the map
    """
    out = []
    for entry, exit_ in trades:
        entry = pd.to_datetime(entry)
        exit_ = pd.to_datetime(exit_) if exit_ is not None else None
        expiry = expiry_map.get(entry)
        if expiry is None:
            # no listed contract starting exactly at 'entry' → strict alignment: skip
            continue
        if exit_ is None or exit_ > expiry:
            exit_capped = expiry
        else:
            exit_capped = exit_
        out.append((entry, exit_capped, expiry))
    return out

def trades_and_mtm_from_adjusted(option_df, adjusted_trades, fv_col="FV", forbid_overlap=True):
    """
    option_df: rows for ONE underlying & ONE range with ['startDate','endDate','date', fv_col, ...]
    adjusted_trades: list of (entry, exit_capped, expiry)
    Returns:
      trades_df and daily MTM series concatenated across trades
    """
    df = option_df.copy()
    df["startDate"] = pd.to_datetime(df["startDate"])
    df["endDate"]   = pd.to_datetime(df["endDate"])
    df["date"]      = pd.to_datetime(df["date"])

    trades_rows, pnl_chunks = [], []
    last_exit = None

    for entry, exit_capped, expiry in adjusted_trades:
        # strict: must match startDate exactly
        sub = df[(df["startDate"] == entry) & (df["endDate"] == expiry)]
        if sub.empty:
            continue

        if forbid_overlap and last_exit is not None and entry <= last_exit:
            continue

        path = sub[(sub["date"] >= entry) & (sub["date"] <= exit_capped)].sort_values("date")
        if path.empty or entry not in set(path["date"]) or exit_capped not in set(path["date"]):
            continue

        entry_fv = float(path.loc[path["date"] == entry, fv_col].iloc[0])
        exit_fv  = float(path.loc[path["date"] == exit_capped, fv_col].iloc[0])
        pnl = exit_fv - entry_fv

        tp = path.copy()
        tp["MTM"] = tp[fv_col] - entry_fv
        pnl_chunks.append(tp[["date","MTM"]])

        trades_rows.append({
            "entry": entry,
            "exit": exit_capped,
            "expiry": expiry,
            "entry_FV": entry_fv,
            "exit_FV": exit_fv,
            "PnL": pnl
        })
        last_exit = exit_capped

    trades_df = pd.DataFrame(trades_rows).sort_values("entry") if trades_rows else pd.DataFrame(
        columns=["entry","exit","expiry","entry_FV","exit_FV","PnL"]
    )
    pnl_ts = (
        pd.concat(pnl_chunks).set_index("date").sort_index()
        if pnl_chunks else pd.DataFrame(columns=["MTM"])
    )
    return trades_df, pnl_ts
----

import pandas as pd

def entries_exits_from_signal(signal_series: pd.Series):
    """
    Strict: détecte les paires (entry_signal_date, exit_signal_date_or_None)
    à partir d'un signal binaire 0/1.
    -> Décale le signal d’un jour pour éviter le look-ahead.
    -> Entry = 0->1 ; Exit = 1->0 (la première après l’entry).
    """
    sig = signal_series.shift(1).fillna(0).astype(int)
    entries = sig[(sig == 1) & (sig.shift(1).fillna(0) == 0)].index
    exits   = sig[(sig == 0) & (sig.shift(1).fillna(0) == 1)].index

    pairs = []
    exit_iter = iter(exits)
    cur_exit = next(exit_iter, None)
    for e in entries:
        while cur_exit is not None and cur_exit <= e:
            cur_exit = next(exit_iter, None)
        pairs.append((e, cur_exit))  # peut être None si pas d’exit après
    return pairs

def build_trades_and_pnl_timeseries_strict(
    signal_series: pd.Series,
    option_df: pd.DataFrame,
    fv_col: str = "FV",
    forbid_overlap: bool = True
):
    """
    STRICT ALIGNMENT:
      - On ne prend un trade que si un contrat existe avec startDate == entry_signal_date
      - Exit = min(exit_signal_date, expiry) ; si pas d’exit, on tient jusqu’à expiry
      - Pas de 'snap' vers un autre startDate.

    option_df (UN seul underlying & range) doit contenir :
      ['startDate','endDate','date', fv_col, ...]
      Pour chaque startDate, exactement un endDate (=expiry).
      'date' couvre toutes les dates de startDate à endDate incluses.

    Retourne:
      trades_df: [entry_signal_date, startDate, expiry, exitDate, entry_FV, exit_FV, PnL]
      pnl_ts:    série daily MTM (index=date, col='MTM') concaténée sur tous les trades
    """
    df = option_df.copy()
    df["startDate"] = pd.to_datetime(df["startDate"])
    df["endDate"]   = pd.to_datetime(df["endDate"])
    df["date"]      = pd.to_datetime(df["date"])

    sig_pairs = entries_exits_from_signal(signal_series)

    trades = []
    pnl_chunks = []
    last_exit_taken = None

    for entry_sig, exit_sig in sig_pairs:
        entry_sig = pd.to_datetime(entry_sig)
        exit_sig  = pd.to_datetime(exit_sig) if exit_sig is not None else None

        # STRICT: il faut un contrat qui commence EXACTEMENT à entry_sig
        sub_all = df[df["startDate"] == entry_sig]
        if sub_all.empty:
            # aucun contrat listé ce jour-là → on ignore ce signal
            continue

        # on suppose un seul expiry pour ce startDate
        expiry = sub_all["endDate"].iloc[0]

        # exit = min(exit_sig, expiry) ; si pas d’exit → expiry
        if exit_sig is None or exit_sig > expiry:
            exit_date = expiry
        else:
            exit_date = exit_sig

        # éviter les overlaps si demandé (une position à la fois)
        if forbid_overlap and last_exit_taken is not None and entry_sig <= last_exit_taken:
            continue

        # chemin quotidien du trade
        trade_path = sub_all[(sub_all["date"] >= entry_sig) & (sub_all["date"] <= exit_date)].sort_values("date")
        if trade_path.empty:
            continue
        # s'assurer qu'on a bien les points d'entrée/sortie
        if entry_sig not in set(trade_path["date"]) or exit_date not in set(trade_path["date"]):
            continue

        entry_fv = float(trade_path.loc[trade_path["date"] == entry_sig, fv_col].iloc[0])
        exit_fv  = float(trade_path.loc[trade_path["date"] == exit_date,   fv_col].iloc[0])
        pnl      = exit_fv - entry_fv

        # MTM quotidien relatif à l'entrée
        tp = trade_path.copy()
        tp["MTM"] = tp[fv_col] - entry_fv

        trades.append({
            "entry_signal_date": entry_sig,
            "startDate": entry_sig,      # strict: start == entry
            "expiry": expiry,
            "exitDate": exit_date,
            "entry_FV": entry_fv,
            "exit_FV": exit_fv,
            "PnL": pnl
        })
        pnl_chunks.append(tp[["date","MTM"]])

        last_exit_taken = exit_date

    trades_df = pd.DataFrame(trades).sort_values("startDate") if trades else pd.DataFrame(
        columns=["entry_signal_date","startDate","expiry","exitDate","entry_FV","exit_FV","PnL"]
    )
    pnl_ts = (
        pd.concat(pnl_chunks).set_index("date").sort_index()
        if pnl_chunks else pd.DataFrame(columns=["MTM"])
    )
    return trades_df, pnl_ts
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
