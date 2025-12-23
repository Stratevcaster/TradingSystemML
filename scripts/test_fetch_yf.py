import yfinance as yf
print('yfinance version:', yf.__version__)
df = yf.download('BTC-USD', start='2020-01-01', end='2025-12-15', interval='1d')
print('rows:', len(df))
print(df.head())
