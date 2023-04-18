import yfinance as yf
import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import csv
import concurrent.futures
import datetime


def stock_rmse(ticker):
    # Get today's date
    today = datetime.datetime.now()

    # Subtract one day to get yesterday's date
    yesterday = today - datetime.timedelta(days=1)

    # Format yesterday's date as a string in yyyy-mm-dd format
    yesterday_str = yesterday.strftime("%Y-%m-%d")

    try:
        data = yf.download(ticker, start="2010-01-01", end=yesterday_str)
    except (KeyError, AttributeError) as error:
        return {'ticker': ticker, 'yhat': [0], 'yhat_lower': [0], 'yhat_upper': [0], 'trend': 0,
                'rmse': 100, 'mape': 100}

    # Prepare data for Prophet model
    df = pd.DataFrame({'ds': data.index[:-30], 'y': data['Adj Close'][:-30]})
    df.reset_index(inplace=True, drop=True)

    # Create and fit the Prophet model
    m = Prophet(daily_seasonality=True)

    try:
        m.fit(df)
    except ValueError:
        return {'ticker': ticker, 'yhat': [0], 'yhat_lower': [0], 'yhat_upper': [0], 'trend': 0,
                'rmse': 100, 'mape': 100}

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


def multi_threaded_stock_rmse(tickers):
    with concurrent.futures.ThreadPoolExecutor(4) as executor:
        results = list(executor.map(stock_rmse, tickers))

    return results


def get_tickers():
    raw_df = pd.read_csv('./folder/raw_tickers.csv')
    ticker_list = raw_df.values.tolist()
    ticker_list = [t[1] for t in ticker_list]
    return ticker_list


def test_csv():
    df = pd.read_csv('test.csv')
    return df


def run():
    ticker_list = get_tickers()

    T_LIST_SIZE = len(ticker_list)
    i = 0
    d_list = []

    for t in ticker_list:
        d = stock_rmse(t)
        d_list.append(d)
        i += 1
        print(f'====================== {(i / T_LIST_SIZE) * 100:4.4}% =====================')

    # d_list = multi_threaded_stock_rmse(ticker_list)

    fd = open('all_out_2.csv', 'w')
    writer = csv.writer(fd)

    writer.writerow(['ticker', 'yhat', 'yhat_lower', 'yhat_upper', 'trend', 'rmse', 'mape'])

    for d in d_list:
        writer.writerow(d.values())

    fd.close()

    # getting lowest rmse

    df = pd.read_csv('all_out_2.csv')
    df.sort_values(by='rmse', inplace=True)

    low_ticks = []

    for d in df.ticker[:10]:
        low_ticks.append(d)

    print(low_ticks)


if __name__ == '__main__':
    run()