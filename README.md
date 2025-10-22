import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

results = {}

for u, df_u in underlier_datasets.items():
    print(f"\n=== {u} ===")

    df_u = df_u.sort_values("trade_date").reset_index(drop=True)
    
    # Create next-day RFQ target
    df_u["target_next_rfqs"] = df_u["rfq_count"].shift(-1)
    df_u = df_u.dropna(subset=["target_next_rfqs"])

    # Select features
    feature_cols = [col for col in df_u.columns if "change" in col]
    
    X = df_u[feature_cols]
    y = df_u["target_next_rfqs"]

    # Time-based split (80/20)
    split_idx = int(len(df_u) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # --- Linear Regression ---
    lin_model = LinearRegression()
    lin_model.fit(X_train, y_train)
    lin_preds = lin_model.predict(X_test)
    lin_r2 = r2_score(y_test, lin_preds)
    lin_mse = mean_squared_error(y_test, lin_preds)

    # --- Random Forest ---
    rf_model = RandomForestRegressor(
        n_estimators=300, max_depth=5, random_state=0
    )
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    rf_r2 = r2_score(y_test, rf_preds)
    rf_mse = mean_squared_error(y_test, rf_preds)

    # Save results
    results[u] = {
        "linear": {"model": lin_model, "r2": lin_r2, "mse": lin_mse},
        "random_forest": {"model": rf_model, "r2": rf_r2, "mse": rf_mse},
    }

    print(f"Linear Regression  -> R² = {lin_r2:.3f}, MSE = {lin_mse:.2f}")
    print(f"Random Forest      -> R² = {rf_r2:.3f}, MSE = {rf_mse:.2f}")



import matplotlib.pyplot as plt
import numpy as np

for u in underlier_datasets.keys():
    rf = results[u]["random_forest"]["model"]
    importances = rf.feature_importances_
    feature_cols = [col for col in underlier_datasets[u].columns if "change" in col]
    indices = np.argsort(importances)[::-1][:10]  # top 10 features
    
    plt.figure(figsize=(8, 4))
    plt.barh(np.array(feature_cols)[indices], importances[indices])
    plt.title(f"Top features for {u}")
    plt.gca().invert_yaxis()
    plt.show()


for u, df_u in underlier_datasets.items():
    df_u = df_u.sort_values("trade_date").reset_index(drop=True)
    feature_cols = [col for col in df_u.columns if "change" in col]
    
    split_idx = int(len(df_u) * 0.8)
    X_test = df_u.iloc[split_idx:][feature_cols]
    y_test = df_u.iloc[split_idx:]["rfq_count"].shift(-1).dropna()
    
    rf_model = results[u]["random_forest"]["model"]
    preds = rf_model.predict(X_test)
    
    plt.figure(figsize=(10, 4))
    plt.plot(df_u.iloc[split_idx:]["trade_date"].iloc[:len(preds)], y_test.values[:len(preds)], label="Actual RFQs", linewidth=2)
    plt.plot(df_u.iloc[split_idx:]["trade_date"].iloc[:len(preds)], preds, "--", label="Predicted RFQs")
    plt.title(f"{u} – Next-Day RFQ Prediction (Random Forest)")
    plt.legend()
    plt.show()
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
