import numpy as np
import pandas as pd

def find_multiple_ranges_for_product(prop_df, edge_min_frac=0.001):
    """
    prop_df : DataFrame avec colonnes
              ['STRATEGY_TYPE','UNDERLYING_INSTRUMENT_TYPE','STRIKE_PCT_SPOT_BIN',
               'TENOR','Proportion']
              --> déjà filtré pour UN produit donné (strategy + underlying)
    edge_min_frac : seuil proportion minimale par case (ex: 0.001 = 0,1%)
    Retourne : liste de dicts { 'strike_min', 'strike_max', 'tenor_min', 'tenor_max', 'mass', 'cells' }
    """
    # Pivot table Tenor x Strike
    pivot = prop_df.pivot_table(
        index="TENOR", columns="STRIKE_PCT_SPOT_BIN", values="Proportion", fill_value=0.0
    )

    tenors = list(pivot.index)
    strikes = list(pivot.columns)
    M = pivot.values.copy()

    visited = np.zeros_like(M, dtype=bool)
    cluster_total = M.sum()

    results = []

    while True:
        # Trouver cellule max non visitée
        M_masked = np.where(visited, -1, M)  # mettre -1 aux visités pour les ignorer
        max_idx = np.unravel_index(np.argmax(M_masked), M.shape)
        max_val = M_masked[max_idx]

        # Stop si plus de case au-dessus du seuil
        if max_val < edge_min_frac:
            break

        # Nouvelle graine
        block_cells = set([max_idx])
        visited[max_idx] = True

        # Frontière pour BFS
        frontier = [max_idx]
        while frontier:
            i, j = frontier.pop()
            # Parcourir les voisins (8 directions)
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < M.shape[0] and 0 <= nj < M.shape[1]:
                        if not visited[ni, nj] and M[ni, nj] >= edge_min_frac:
                            visited[ni, nj] = True
                            block_cells.add((ni, nj))
                            frontier.append((ni, nj))

        # Calculer range strike/tenor pour ce bloc
        rows = [c[0] for c in block_cells]
        cols = [c[1] for c in block_cells]
        strike_min = strikes[min(cols)]
        strike_max = strikes[max(cols)]
        tenor_min = tenors[min(rows)]
        tenor_max = tenors[max(rows)]

        mass = sum(M[i, j] for i, j in block_cells)

        results.append({
            "strike_min": strike_min,
            "strike_max": strike_max,
            "tenor_min": tenor_min,
            "tenor_max": tenor_max,
            "mass": mass,
            "cells": [(tenors[i], strikes[j]) for i, j in block_cells]
        })

    return results

------
import pandas as pd
import numpy as np

# ---- robust sort helpers ----
def _strike_key(x):
    s = str(x).strip().replace('%','').replace(' ','').replace('+','')
    if '-' in s:
        a, _ = s.split('-', 1)
        try: return float(a)
        except: return float('inf')
    try: return float(s)
    except: return float('inf')

def _tenor_key(x):
    s = str(x).strip().lower().replace(' ','').replace('m','')
    try:
        v = float(s)
        return v/30.0 if v > 24 else v  # rough days->months
    except:
        return float('inf')

def _build_matrix(sub, strike_col, tenor_col, prop_col):
    strikes = sorted(sub[strike_col].dropna().unique(), key=_strike_key)
    tenors  = sorted(sub[tenor_col].dropna().unique(),  key=_tenor_key)
    if len(strikes)==0 or len(tenors)==0:
        return strikes, tenors, np.zeros((0,0)), None
    pv = (sub.pivot_table(index=tenor_col, columns=strike_col, values=prop_col,
                          aggfunc=np.sum, fill_value=0.0)
            .reindex(index=tenors, columns=strikes, fill_value=0.0))
    return strikes, tenors, pv.values.astype(float), pv  # keep pivot for labels if needed

def _neighbors8(mask, r, c):
    R, C = mask.shape
    for dr in (-1,0,1):
        for dc in (-1,0,1):
            if dr==0 and dc==0: 
                continue
            rr, cc = r+dr, c+dc
            if 0 <= rr < R and 0 <= cc < C:
                yield rr, cc

def _grow_connected_area(M, edge_min_abs):
    """
    Greedy growth:
      - seed = argmax cell
      - repeatedly add the *neighbor* (8-neighborhood) with max mass >= edge_min_abs
      - returns selected mask and total mass
    """
    R, C = M.shape
    if R==0 or C==0: 
        return np.zeros((0,0), dtype=bool), 0.0

    sel = np.zeros((R,C), dtype=bool)
    r0, c0 = np.unravel_index(np.argmax(M), M.shape)
    if M[r0, c0] <= 0:
        return sel, 0.0

    sel[r0, c0] = True
    total = float(M[r0, c0])

    # frontier = all neighbors of selected set not yet selected
    while True:
        best_gain = 0.0
        best_rc = None

        # collect unique neighbors
        seen = set()
        rs, cs = np.where(sel)
        for r, c in zip(rs, cs):
            for rr, cc in _neighbors8(sel, r, c):
                if sel[rr, cc]: 
                    continue
                key = (rr, cc)
                if key in seen: 
                    continue
                seen.add(key)
                gain = float(M[rr, cc])
                if gain > best_gain:
                    best_gain = gain
                    best_rc = key

        if best_rc is None or best_gain < edge_min_abs:
            break

        rr, cc = best_rc
        sel[rr, cc] = True
        total += best_gain

    return sel, total

def find_product_ranges_connected(
    proportions_df,
    edge_min_cluster_frac=0.001,  # 0.1% of cluster
    group_cols=("STRATEGY_TYPE","UNDERLYING_INSTRUMENT_TYPE"),
    strike_col="STRIKE_PCT_SPOT_BIN",
    tenor_col="TENOR",
    prop_col="Proportion"
):
    """
    For each product (strategy, underlying type):
      - build the strike×tenor mass matrix in *absolute cluster* units (no local normalization)
      - seed at max cell, grow with 8-neighborhood, adding neighbors only if gain >= edge_min_cluster_frac * cluster_total
      - return one range: [min_strike..max_strike] × [min_tenor..max_tenor]
      - report PRODUCT_RANGE_FRAC and CLUSTER_FRAC
    """
    # cluster total (after any prior filtering applied to proportions_df)
    cluster_total = float(proportions_df[prop_col].sum())
    if cluster_total <= 0:
        # empty input
        return pd.DataFrame(columns=list(group_cols)+[
            "STRIKE_START","STRIKE_END","TENOR_START","TENOR_END",
            "CLUSTER_FRAC","PRODUCT_RANGE_FRAC","PRODUCT_TOTAL_FRAC",
            "CELLS_SELECTED"
        ])

    edge_min_abs = edge_min_cluster_frac * cluster_total

    rows = []
    for keys, sub in proportions_df.groupby(list(group_cols)):
        strikes, tenors, M, pv = _build_matrix(sub, strike_col, tenor_col, prop_col)
        if M.size == 0:
            rows.append(dict(zip(group_cols, keys), **{
                "STRIKE_START": None, "STRIKE_END": None,
                "TENOR_START": None,  "TENOR_END": None,
                "CLUSTER_FRAC": 0.0,
                "PRODUCT_RANGE_FRAC": 0.0,
                "PRODUCT_TOTAL_FRAC": 0.0,
                "CELLS_SELECTED": []
            }))
            continue

        product_total = float(M.sum())  # absolute mass of this product in cluster
        if product_total <= 0:
            rows.append(dict(zip(group_cols, keys), **{
                "STRIKE_START": None, "STRIKE_END": None,
                "TENOR_START": None,  "TENOR_END": None,
                "CLUSTER_FRAC": 0.0,
                "PRODUCT_RANGE_FRAC": 0.0,
                "PRODUCT_TOTAL_FRAC": 0.0,
                "CELLS_SELECTED": []
            }))
            continue

        sel_mask, sel_mass = _grow_connected_area(M, edge_min_abs=edge_min_abs)

        if sel_mask.sum() == 0:
            # nothing passes threshold → take only the top cell (ensures at least one)
            r0, c0 = np.unravel_index(np.argmax(M), M.shape)
            sel_mask = np.zeros_like(M, dtype=bool)
            sel_mask[r0, c0] = True
            sel_mass = float(M[r0, c0])

        # derive strike/tenor ranges from selected cells
        rs, cs = np.where(sel_mask)
        t_start = tenors[int(rs.min())]
        t_end   = tenors[int(rs.max())]
        k_start = strikes[int(cs.min())]
        k_end   = strikes[int(cs.max())]

        # list selected label pairs (optional, for debugging/explainability)
        cells = [(tenors[int(r)], strikes[int(c)], float(M[int(r), int(c)])) for r, c in zip(rs, cs)]

        rows.append(dict(zip(group_cols, keys), **{
            "STRIKE_START": k_start,
            "STRIKE_END":   k_end,
            "TENOR_START":  t_start,
            "TENOR_END":    t_end,
            # share of *cluster* covered by this product's range:
            "CLUSTER_FRAC": float(sel_mass / cluster_total),
            # share of *this product* covered by the range:
            "PRODUCT_RANGE_FRAC": float(sel_mass / product_total),
            # share of cluster that this whole product represents (context):
            "PRODUCT_TOTAL_FRAC": float(product_total / cluster_total),
            "CELLS_SELECTED": cells
        }))

    out = pd.DataFrame(rows).sort_values(list(group_cols)).reset_index(drop=True)
    return out


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
    
