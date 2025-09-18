
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
