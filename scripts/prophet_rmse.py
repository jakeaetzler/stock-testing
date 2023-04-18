import yfinance as yf
import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import csv

def stock_rmse(ticker):
    data = yf.download(ticker, start="2010-01-01", end="2023-04-01")

    # Prepare data for Prophet model
    df = pd.DataFrame({'ds': data.index[:-30], 'y': data['Adj Close'][:-30]})
    df.reset_index(inplace=True, drop=True)

    # Create and fit the Prophet model
    m = Prophet(daily_seasonality=True)

    try:
        m.fit(df)
    except ValueError:
        return (ticker, 100)


    # Generate predictions for the next 30 days
    future = m.make_future_dataframe(periods=30)
    forecast = m.predict(future)

    # Extract summary statistics from forecast DataFrame
    yhat = forecast['yhat']
    yhat_lower = forecast['yhat_lower']
    yhat_upper = forecast['yhat_upper']
    trend = forecast['trend']

    actual = data['Adj Close'][-30:]
    predicted = forecast['yhat'][-30:]
    mape = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)


    s_dict = {'ticker': ticker, 'yhat': yhat, 'yhat_lower': yhat_lower, 'yhat_upper': yhat_upper, 'trend': trend,
              'rmse': rmse, 'mape': mape}

    return s_dict
