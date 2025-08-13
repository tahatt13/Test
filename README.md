# Test



import pandas as pd
import numpy as np

# --- helpers tri robustes ---
def safe_strike_sort_key(x):
    s = str(x).strip().replace('%','').replace(' ','').replace('+','')
    if '-' in s:
        a, _ = s.split('-', 1)
        try: return float(a)
        except: return float('inf')
    try: return float(s)
    except: return float('inf')

def safe_tenor_sort_key(x):
    s = str(x).strip().lower().replace(' ','').replace('m','')
    try:
        v = float(s)
        return v/30.0 if v > 24 else v  # ~jours→mois
    except:
        return float('inf')

# --- construit la matrice normalisée (tenor x strike) pour un groupe ---
def _group_matrix(sub, strike_col="STRIKE_PCT_SPOT_BIN", tenor_col="TENOR", prop_col="Proportion"):
    strikes = sorted(sub[strike_col].dropna().unique(), key=safe_strike_sort_key)
    tenors  = sorted(sub[tenor_col].dropna().unique(),  key=safe_tenor_sort_key)
    if len(strikes)==0 or len(tenors)==0:
        return strikes, tenors, np.zeros((0,0), dtype=float)

    pivot = (sub.pivot_table(index=tenor_col, columns=strike_col, values=prop_col,
                             aggfunc=np.sum, fill_value=0.0)
               .reindex(index=tenors, columns=strikes, fill_value=0.0))
    M = pivot.values.astype(float)
    tot = M.sum()
    if tot > 0: M = M / tot
    return strikes, tenors, M

# --- fait croître un rectangle autour d'une graine (r,c) ---
def _grow_rectangle_from_seed(M, r_init, c_init, edge_min=0.005, mass_goal=None):
    R, C = M.shape
    if R==0 or C==0 or M[r_init, c_init] <= 0: return None
    r0 = r1 = int(r_init); c0 = c1 = int(c_init)
    mass = float(M[r0, c0])

    def row_mass(r, c0, c1): return float(M[r, c0:c1+1].sum())
    def col_mass(c, r0, r1): return float(M[r0:r1+1, c].sum())

    while True:
        candidates = []
        if r0-1 >= 0:   candidates.append(("up",    row_mass(r0-1, c0, c1)))
        if r1+1 < R:    candidates.append(("down",  row_mass(r1+1, c0, c1)))
        if c0-1 >= 0:   candidates.append(("left",  col_mass(c0-1, r0, r1)))
        if c1+1 < C:    candidates.append(("right", col_mass(c1+1, r0, r1)))
        if not candidates: break

        # prend l’extension la plus “rentable”
        direction, gain = max(candidates, key=lambda x: x[1])
        if gain < edge_min: break
        if (mass_goal is not None) and (mass >= mass_goal): break

        if direction == "up":    r0 -= 1
        elif direction == "down":r1 += 1
        elif direction == "left":c0 -= 1
        else:                    c1 += 1
        mass += gain

    return (r0, r1, c0, c1, float(mass))

# --- rectangles 2D depuis top 'seed_k' couples, jusqu’à 'top_k' rects et 'mass_target' ---
def find_2d_product_ranges_seeds(proportions_df,
                                 mass_target=0.8,
                                 top_k=2,
                                 seed_k=3,
                                 edge_min=0.005,
                                 group_cols=("STRATEGY_TYPE","UNDERLYING_INSTRUMENT_TYPE"),
                                 strike_col="STRIKE_PCT_SPOT_BIN",
                                 tenor_col="TENOR",
                                 prop_col="Proportion"):
    """
    Pour chaque (STRATEGY_TYPE, UNDERLYING_INSTRUMENT_TYPE):
      1) construit la matrice Proportion (TENOR x STRIKE), normalisée
      2) prend les 'seed_k' cellules les plus massives comme graines
      3) pour chaque graine (ordre décroissant), fait croître un rectangle 2D
         en ajoutant lignes/colonnes si gain >= edge_min
      4) met à zéro le rectangle retenu et cumule jusqu’à mass_target ou top_k rectangles
    """
    rows = []

    for keys, sub in proportions_df.groupby(list(group_cols)):
        strikes, tenors, M = _group_matrix(sub, strike_col, tenor_col, prop_col)
        if M.size == 0 or M.sum() <= 0:
            rows.append(dict(zip(group_cols, keys), **{
                "block_id": None, "STRIKE_START": None, "STRIKE_END": None,
                "TENOR_START": None, "TENOR_END": None, "MASS": 0.0, "CUM_MASS": 0.0
            }))
            continue

        flat = M.ravel()
        order = np.argsort(-flat)  # indices triés par masse décroissante
        seeds = []
        seen = set()
        for idx in order:
            if flat[idx] <= 0: break
            r, c = divmod(idx, M.shape[1])
            if (r,c) in seen: 
                continue
            seeds.append((r,c))
            seen.add((r,c))
            if len(seeds) >= seed_k: 
                break

        Mwork = M.copy()
        cum_mass = 0.0
        block_id = 0

        # 1er passage: utiliser les graines choisies
        for (r_seed, c_seed) in seeds:
            if block_id >= top_k or cum_mass >= mass_target: 
                break
            if Mwork[r_seed, c_seed] <= 0: 
                continue
            rect = _grow_rectangle_from_seed(Mwork, r_seed, c_seed, edge_min=edge_min,
                                             mass_goal=(mass_target - cum_mass))
            if rect is None: 
                continue
            r0, r1, c0, c1, m = rect
            block_id += 1
            cum_mass += m
            rows.append(dict(zip(group_cols, keys), **{
                "block_id": block_id,
                "STRIKE_START": strikes[c0],
                "STRIKE_END":   strikes[c1],
                "TENOR_START":  tenors[r0],
                "TENOR_END":    tenors[r1],
                "MASS": float(m),
                "CUM_MASS": float(cum_mass)
            }))
            # vider le rectangle
            Mwork[r0:r1+1, c0:c1+1] = 0.0

        # 2e passage (optionnel): si on n’a pas atteint top_k ou mass_target,
        # continuer avec les meilleures cellules restantes
        while (block_id < top_k) and (cum_mass < mass_target) and (Mwork.sum() > 0):
            r_seed, c_seed = np.unravel_index(np.argmax(Mwork), Mwork.shape)
            if Mwork[r_seed, c_seed] <= 0:
                break
            rect = _grow_rectangle_from_seed(Mwork, r_seed, c_seed, edge_min=edge_min,
                                             mass_goal=(mass_target - cum_mass))
            if rect is None:
                break
            r0, r1, c0, c1, m = rect
            block_id += 1
            cum_mass += m
            rows.append(dict(zip(group_cols, keys), **{
                "block_id": block_id,
                "STRIKE_START": strikes[c0],
                "STRIKE_END":   strikes[c1],
                "TENOR_START":  tenors[r0],
                "TENOR_END":    tenors[r1],
                "MASS": float(m),
                "CUM_MASS": float(cum_mass)
            }))
            Mwork[r0:r1+1, c0:c1+1] = 0.0

        # si rien trouvé, renvoyer une ligne vide lisible
        if block_id == 0:
            rows.append(dict(zip(group_cols, keys), **{
                "block_id": None,
                "STRIKE_START": None, "STRIKE_END": None,
                "TENOR_START": None,  "TENOR_END": None,
                "MASS": 0.0, "CUM_MASS": 0.0
            }))

    out = pd.DataFrame(rows).sort_values(list(group_cols)+["block_id"]).reset_index(drop=True)
    return out
    
