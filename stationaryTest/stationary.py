import yfinance as yf
import pandas as pd
import numpy as np
from fracdiff.sklearn import Fracdiff
from statsmodels.tsa.stattools import adfuller, kpss

# Configuration and Parameters
start_date = "2014-01-01"
end_date = "2024-01-01"
symbols = ["AAPL", "IBM"]
window_length = 30
d_values = np.arange(0.01, 1.1, 0.01)

# Downloading stock data
data = {}
for symbol in symbols:
    try:
        data[symbol] = yf.download(symbol, start=start_date, end=end_date)["Close"]
        print(f"Data downloaded for {symbol}")
    except Exception as e:
        print(f"Failed to download data for {symbol}: {e}")


# Function to apply statistical tests
def apply_tests(data, lag="AIC", regression_type="c"):
    adf_result = adfuller(data, autolag=lag)
    kpss_result = kpss(data, regression=regression_type, nlags="auto")
    return (adf_result[1], kpss_result[1], adf_result[1] < 0.05, kpss_result[1] > 0.05)


# Preparing and analyzing data
optimal_d = {}
detailed_results = []

for symbol, series in data.items():
    print(f"Analyzing {symbol}")
    for d_value in d_values:
        fracdiff = Fracdiff(d_value)
        adf_p_values = []
        kpss_p_values = []
        valid_windows = 0
        passing_windows = 0

        for i in range(len(series) - window_length + 1):
            window = series[i : i + window_length]
            try:
                transformed = fracdiff.fit_transform(
                    window.values.reshape(-1, 1)
                ).flatten()
                adf_p, kpss_p, adf_pass, kpss_pass = apply_tests(transformed)
                adf_p_values.append(adf_p)
                kpss_p_values.append(kpss_p)
                valid_windows += 1
                if adf_pass and kpss_pass:
                    passing_windows += 1
            except Exception as e:
                print(f"Error processing window at index {i}: {e}")

        if valid_windows > 0:
            pass_ratio = passing_windows / valid_windows
            detailed_results.append(
                {
                    "Symbol": symbol,
                    "D_value": d_value,
                    "Mean_ADF_p_value": np.mean(adf_p_values),
                    "Mean_KPSS_p_value": np.mean(kpss_p_values),
                    "STD_ADF_p_value": np.std(adf_p_values),
                    "STD_KPSS_p_value": np.std(kpss_p_values),
                    "Pass_Ratio": pass_ratio,
                    "Valid_Windows": valid_windows,
                }
            )

# Dataframe and Output
df = pd.DataFrame(detailed_results)
df.to_csv("stock_analysis_results.csv", index=False)
for symbol in symbols:
    symbol_df = df[df["Symbol"] == symbol]
    first_passing_d = symbol_df[symbol_df["Pass_Ratio"] == 1.0]["D_value"].min()
    optimal_d[symbol] = first_passing_d if not np.isnan(first_passing_d) else None
    print(f"Optimal d for {symbol} (first d with 100% pass ratio): {optimal_d[symbol]}")

# Final Output
print("Optimal differencing degrees for stationarity:")
for symbol, d in optimal_d.items():
    if d is not None:
        print(f"{symbol}: d={d:.2f}")
    else:
        print(
            f"{symbol}: No d value found that achieves full stationarity across all windows."
        )
