import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
from pmdarima import auto_arima
from pandas.plotting import autocorrelation_plot

# csv data from the identity theft resource center
file_path = 'compromises.csv'
df = pd.read_csv(file_path)

# convert the year column to a pandas datetime object and set it as the index
df['Year'] = pd.to_datetime(df['Year'], format='%Y')
df.set_index('Year', inplace=True)
df = df.sort_index()

# add a frequency to the index
df.index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='AS-JAN')

# log transformation to stabilize variance
df['Log_Compromises'] = np.log1p(df['Compromises'])

# time series
log_compromises = df['Log_Compromises']

# autocorrelation for log-trransformed compromises
autocorrelation_plot(log_compromises)
plt.title("Autocorrelation for Log(Compromises)")
plt.show()

# auto_arima to find the best (p, d, q) combination
arima_model = auto_arima(
    log_compromises,
    seasonal=False, 
    stepwise=True,
    trace=True,
    error_action='ignore',
    suppress_warnings=True
)

print(arima_model.summary())

# forecast
forecast_years = pd.date_range(start='2024', periods=5, freq='YE')
log_forecast = arima_model.predict(n_periods=5)

# revert forecast from the log transformation for visualizations
forecast = np.expm1(log_forecast)

# evaluation using a test and train split
train = log_compromises[:-5]
test = log_compromises[-5:]

# fit on training data
arima_train_model = auto_arima(
    train,
    seasonal=False,
    stepwise=True,
    trace=False,
    error_action='ignore',
    suppress_warnings=True
)

# forecast test
log_predictions = arima_train_model.predict(n_periods=5)

# revert predictions back to original from log transformation
predictions = np.expm1(log_predictions)

# evaluation of rmse
rmse = sqrt(mean_squared_error(np.expm1(test), predictions))
print(f'Compromises RMSE: {rmse}')


forecast_df = pd.DataFrame({
    'Year': forecast_years.year,
    'Forecasted Compromises': forecast
})

print("Forecasted Compromises for Next 5 Years:")
print(forecast_df)

# results and visualizations
plt.figure(figsize=(12, 8))

# historical data
plt.plot(df['Compromises'], label='Historical Compromises')

# test data
plt.plot(test.index, np.expm1(test), label='Test Data', linestyle='--')

# predictions by ARIMA
plt.plot(test.index, predictions, label='ARIMA Predictions', linestyle='--')

# the forecasts
plt.plot(forecast_years, forecast, label='Forecasted Compromises', linestyle='--')

# visualizations
plt.title('Compromises Forecast (Log Transformation & ARIMA)')
plt.xlabel('Year')
plt.ylabel('Compromises')
plt.legend()

plt.tight_layout()
plt.show()

# evaluation of the model
print("Model Evaluation:")
print(arima_train_model.summary())
print(f'Forecasted Compromises for Next 5 Years:')

print(forecast_df)
def forecast_metrics(y_true, y_pred):
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    min_val = np.min(y_pred)
    max_val = np.max(y_pred)
    return mape, mae, rmse, min_val, max_val

mape, mae, rmse, min_val, max_val = forecast_metrics(np.expm1(test), predictions)
print(f'MAPE: {mape}')
print(f'MAE: {mae}')
print(f'RMSE: {rmse}')
print(f'Min Value: {min_val}')
print(f'Max Value: {max_val}')


