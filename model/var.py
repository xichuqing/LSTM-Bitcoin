# %%
from statsmodels.tsa.api import VAR
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint
from pathlib import Path
import numpy as np


target_column = 'Close'
predictors = ['Close_Diff', 'polarity_Diff'] 

root = Path(__file__).resolve().parent.parent
data_path = root / 'data' / 'df.csv'
df = pd.read_csv(data_path)
df.set_index('Date', inplace=True)


# Check for cointegration
coint_test = coint(df['Close'], df['polarity'])
print(f"Cointegration test p-value: {coint_test[1]:.4f}")
if coint_test[1] > 0.05:
    print("The series are not cointegrated. VECM may not be appropriate. We will use VAR")
else:
    print("The series are cointegrated. Proceeding with VECM.")


# Check stationarity of the series
def check_stationarity(series, significance_level=0.05):
    result = adfuller(series.dropna())
    p_value = result[1]
    if p_value < significance_level:
        print(f"Series is stationary (p-value: {p_value:.4f})")
        return True
    else:
        print(f"Series is non-stationary (p-value: {p_value:.4f})")
        return False

d = 0
df['Close_Diff'] = df['Close']
while not check_stationarity(df['Close_Diff']):
    d += 1
    df['Close_Diff'] = df['Close_Diff'].diff().dropna()
    print(f"Differenced {d} time(s)")
print(f"Total differences needed: {d}")

d = 0
df['polarity_Diff'] = df['polarity']
while not check_stationarity(df['polarity_Diff']):
    d += 1
    df['polarity_Diff'] = df['polarity_Diff'].diff().dropna()
    print(f"Differenced {d} time(s)")
print(f"Total differences needed: {d}")
df.dropna(inplace=True)

var_df = df[predictors]
split_idx = int(len(var_df) * 0.7)
var_train, var_test = var_df[:split_idx], var_df[split_idx:]

# Initialize lists to store results
all_forecasts = []
all_actuals = []

# Recursive VAR model
for i in range(0, len(var_test), 7):  # Update every 30 days (1 month)
    # Calculate the number of steps to forecast
    steps = min(7, len(var_test) - i)  # Ensure we don't exceed the test data length
    # Update training data
    var_train = var_df[:split_idx + i]
    # Fit VAR model
    model = VAR(var_train)
    results = model.select_order(maxlags=15)
    model_fitted = model.fit(maxlags=15, ic='aic')
    # Forecast next steps
    lag_order = model_fitted.k_ar
    forecast_input = var_train.values[-lag_order:]
    forecast = model_fitted.forecast(y=forecast_input, steps=steps)
    forecast_df = pd.DataFrame(forecast, index=var_test.index[i:i+steps], columns=predictors)
    # Reconstruct Close price
    last_close = df['Close'].iloc[split_idx + i - 1]
    forecast_close = forecast_df['Close_Diff'].cumsum() + last_close
    # Store results
    all_forecasts.extend(forecast_close)
    all_actuals.extend(df['Close'].iloc[split_idx + i:split_idx + i + steps])

# Convert results to DataFrame
results_df = pd.DataFrame({
    'Date': var_test.index[:len(all_forecasts)],
    'Actual Close': all_actuals,
    'Forecasted Close': all_forecasts
})
# %%
plt.figure(figsize=(10, 6))
plt.plot(results_df['Date'], results_df['Actual Close'], label='Actual Close Price', marker='o')
plt.plot(results_df['Date'], results_df['Forecasted Close'], label='Forecasted Close Price', linestyle='--', marker='x')
plt.title('Actual vs Forecasted Close Price (Recursive VAR)')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.grid()
plt.show()

# Evaluate the model
r2_close = r2_score(results_df['Actual Close'], results_df['Forecasted Close'])
mae_close = mean_absolute_error(results_df['Actual Close'], results_df['Forecasted Close'])
mse_close = mean_squared_error(results_df['Actual Close'], results_df['Forecasted Close'])
print(f'R² for Reconstructed Close Price: {r2_close:.4f}')
print(f'MAE for Reconstructed Close Price: {mae_close:.4f}')
print(f'MSE for Reconstructed Close Price: {mse_close:.4f}')
# %%