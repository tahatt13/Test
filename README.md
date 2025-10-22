import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

results_classif = {}

for u, df_u in underlier_datasets.items():
    print(f"\n=== {u} ===")

    # Sort by date to ensure correct order
    df_u = df_u.sort_values("trade_date").reset_index(drop=True)
    
    # Binary target: will there be any RFQs tomorrow?
    df_u["rfq_active_tomorrow"] = (df_u["rfq_count"].shift(-1) > 0).astype(int)
    
    # Add today’s RFQ count as a feature
    df_u["rfq_count_today"] = df_u["rfq_count"]
    
    # Drop last row (no label due to shift)
    df_u = df_u.dropna(subset=["rfq_active_tomorrow"])
    
    # Select feature columns: all "change ratio" + rfq_count_today
    feature_cols = [col for col in df_u.columns if "change" in col] + ["rfq_count_today"]
    X = df_u[feature_cols]
    y = df_u["rfq_active_tomorrow"]

    # Time-based split (80/20)
    split_idx = int(len(df_u) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Train Random Forest Classifier
    clf = RandomForestClassifier(
        n_estimators=300, 
        max_depth=5, 
        random_state=42,
        class_weight="balanced"   # helps with imbalance (many zeros)
    )
    clf.fit(X_train, y_train)

    # Predictions
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"Accuracy: {acc:.3f} | F1-score: {f1:.3f} | AUC: {auc:.3f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Feature importance
    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1][:10]
    top_feats = np.array(feature_cols)[sorted_idx]
    top_imps = importances[sorted_idx]

    plt.figure(figsize=(8, 4))
    plt.barh(top_feats[::-1], top_imps[::-1])
    plt.title(f"{u} – Top Features Predicting Tomorrow’s RFQ Activity")
    plt.xlabel("Feature Importance")
    plt.show()

    results_classif[u] = {
        "model": clf,
        "accuracy": acc,
        "f1": f1,
        "auc": auc
    }
-------
import numpy as np
import pandas as pd

def getRVData(end_date, horizon, price_data):
    """
    Compute realized volatility (RV) time series for different intervals.
    RV at date t is computed on [t - interval, t - 1].

    Parameters
    ----------
    end_date : pd.Timestamp
        Last date to consider.
    horizon : str
        Horizon for which to compute RVs ("1y", "6m", etc.); parsed by pd.DateOffset.
    price_data : pd.DataFrame
        Must contain columns ["DATE", "CLOSE", "PREVADJCLOSE"], for a single symbol.

    Returns
    -------
    rv_dict : dict
        Keys = {"1m", "3m", "12m"}; values = DataFrames with DATE as index and RV as column.

    Notes
    -----
    • `price_data` must contain sufficient history to cover the requested horizon
      **plus** the maximum interval length.
      For example, if `horizon="1y"` and the longest interval is 252 BD (~1y),
      then `price_data` should cover at least 2 years before `end_date`.
    • If insufficient history is available for a given date/interval,
      the corresponding RV will be `NaN`.
    """

    intervals = {"1m": 22, "3m": 66, "12m": 252}
    rv_dict = {}

    # Copy and ensure DATE is datetime
    price_data = price_data.copy()
    price_data["DATE"] = pd.to_datetime(price_data["DATE"])

    # Compute start_date from horizon
    if horizon.endswith("y"):
        n = int(horizon[:-1])
        start_date = end_date - pd.DateOffset(years=n)
    elif horizon.endswith("m"):
        n = int(horizon[:-1])
        start_date = end_date - pd.DateOffset(months=n)
    else:
        # fallback: interpret as days
        n = int(horizon[:-1]) if horizon.endswith("d") else int(horizon)
        start_date = end_date - pd.Timedelta(days=n)

    # Business day evaluation grid
    date_range = pd.date_range(start=start_date, end=end_date, freq="B")

    # Compute RV for each interval
    for label, interval in intervals.items():
        rv_series = []

        for t in date_range:
            start_window = t - pd.tseries.offsets.BDay(interval)
            end_window = t - pd.tseries.offsets.BDay(1)

            hist_data = price_data.loc[
                (price_data["DATE"] >= start_window) & (price_data["DATE"] <= end_window)
            ]

            if hist_data.empty:
                rv_series.append(np.nan)
            else:
                log_returns = np.log(
                    hist_data["CLOSE"].astype(float) / hist_data["PREVADJCLOSE"].astype(float)
                )
                rv_series.append(np.std(log_returns) * np.sqrt(252))

        rv_dict[label] = pd.DataFrame({"DATE": date_range, "RV": rv_series}).set_index("DATE")

    return rv_dict
