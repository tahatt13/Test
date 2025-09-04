def pick_exit_fv(trade_path, target_date, expiry, fv_col, rule="last"):
    """
    trade_path: df already filtered to this contract, sorted by 'date'
    target_date: desired exit date (for expiry use expiry-1BD)
    expiry: actual expiry date (datetime)
    fv_col: column with FV
    rule: 'last' (<= target), 'next' (>= target), or 'nearest'
    """
    # hard cap: we never read beyond business day before expiry
    cap = (expiry - pd.offsets.BDay(1)).normalize() if isinstance(expiry, pd.Timestamp) else expiry
    target = min(target_date, cap)

    before = trade_path.loc[trade_path["date"] <= target]
    after  = trade_path.loc[trade_path["date"] >= target]

    if rule == "last":
        row = before.tail(1)
        if row.empty:
            # no data before target → fall back to first after (if any), still ≤ cap
            row = after.head(1)
    elif rule == "next":
        row = after.head(1)
        if row.empty:
            # no data after target → fall back to last before
            row = before.tail(1)
    elif rule == "nearest":
        # compare gaps and pick the closer one
        prev = before.tail(1)
        nxt  = after.head(1)
        if prev.empty: row = nxt
        elif nxt.empty: row = prev
        else:
            dprev = (target - prev["date"].iloc[0]).days
            dnxt  = (nxt["date"].iloc[0] - target).days
            row = prev if dprev <= dnxt else nxt
    else:
        raise ValueError("rule must be 'last', 'next', or 'nearest'")

    if row.empty:
        return None  # caller can skip the trade or log a warning
    return float(row[fv_col].iloc[0])



# build full life path for the contract (not just to exit), sorted
trade_path = sub[(sub["date"] >= entry) & (sub["date"] <= expiry)].sort_values("date")
if trade_path.empty:
    print(f"[SKIP] no path for {entry}→{expiry}")
    continue

# entry FV: prefer exact entry, else first >= entry
row_entry = trade_path.loc[trade_path["date"] == entry]
if row_entry.empty:
    row_entry = trade_path.loc[trade_path["date"] >= entry].head(1)
if row_entry.empty:
    print(f"[SKIP] no entry FV at/after {entry}")
    continue
entry_fv = float(row_entry[fv_col].iloc[0])

# target date for exit
target_date = exit_ if exit_ < expiry else (expiry - pd.offsets.BDay(1))

# choose rule: 'last' (common), or 'next' if you want to realize PnL at next available valuation
exit_fv = pick_exit_fv(trade_path, target_date, expiry, fv_col, rule="last")
if exit_fv is None:
    print(f"[SKIP] no exit FV around {target_date} (cap {expiry})")
    continue

