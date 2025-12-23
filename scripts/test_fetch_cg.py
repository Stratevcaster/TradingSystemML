from pycoingecko import CoinGeckoAPI
import pandas as pd
cg = CoinGeckoAPI()
chart = cg.get_coin_market_chart_by_id('bitcoin', vs_currency='usd', days='365')
prices = chart.get('prices', [])
print('len prices', len(prices))
if prices:
    import numpy as np
    df = pd.DataFrame(prices, columns=['date_ms','adjclose'])
    df['date'] = pd.to_datetime(df['date_ms'], unit='ms')
    df.set_index('date', inplace=True)
    print(df[['adjclose']].head())
