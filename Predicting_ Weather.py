import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose

# Simulating some data for renewable energy market (e.g., solar energy prices)
np.random.seed(0)
dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
data = np.random.normal(100, 20, size=len(dates))  # Simulated price data
data = pd.Series(data, index=dates).cumsum() + 1000  # Cumulative sum to simulate price trend
data += np.sin(np.linspace(0, 20, len(dates))) * 50  # Adding seasonality

# Plotting the simulated data
plt.figure(figsize=(10, 5))
plt.plot(data)
plt.title('Simulated Renewable Energy Market Data (e.g., Solar Energy Prices)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

# Decomposing the time series data to analyze trends, seasonality, and residuals
decomposition = seasonal_decompose(data, model='additive', period=365)
fig = decomposition.plot()
fig.set_size_inches(10, 8)
plt.show()

# Fitting a SARIMA model for forecasting
model = SARIMAX(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
results = model.fit()

# Forecasting the next 365 days
forecast = results.get_forecast(steps=365)
forecast_index = pd.date_range(start=data.index[-1], periods=366, closed='right')
forecast_values = forecast.predicted_mean
forecast_conf_int = forecast.conf_int()

# Plotting the forecast
plt.figure(figsize=(10, 5))
plt.plot(data, label='Historical Data')
plt.plot(forecast_index, forecast_values, label='Forecast')
plt.fill_between(forecast_index, forecast_conf_int.iloc[:, 0], forecast_conf_int.iloc[:, 1], color='k', alpha=.15)
plt.title('SARIMA Model Forecast for Next Year')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
