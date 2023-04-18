import yfinance as yf

def get_stock_data():
    stock_list = ['AMZN']
    print('stock_list:', stock_list)
    data = yf.download(stock_list, start="2015-01-01", end="2020-02-21")
    print('data fields downloaded:', set(data.columns.get_level_values(0)))
    return data

print(get_stock_data())