import math
import numpy as np
import pandas as pd

data = pd.read_csv(r"C:\Users\Never nervius\Desktop\18 years\NASDAQ.csv")

prices = data.drop(['Volume'], axis=1)

close = prices['Close']

# Moving Average method
n = 10
prices['SMA'] = prices['Close'].rolling(window=n).mean()

# Weighted 14day Moving Average
n = 10
weights = list(range(1, n + 1))
weights_sum = sum(weights)
prices['WMA'] = close.rolling(window=n).apply(lambda x: (weights * x).sum() / weights_sum, raw=True)

# Momentum
prices['Momentum'] = prices['Close'] - prices['Close'].shift(n)


# Stochastic K% indicator
def stochastic_k(close, low, high, n):
    ll = low.rolling(window=n).min()
    hh = high.rolling(window=n).max()
    return (close - ll) / (hh - ll) * 100


prices['StochasticK'] = stochastic_k(prices['Close'], prices['Low'], prices['High'], n)

# Stochastic D% indicator
stoch_k_values = prices['StochasticK']
stoch_d_values = stoch_k_values.rolling(n).mean() * 100
prices['Stochastic_D'] = stoch_d_values

#RSI
def rsi(price, n):
    deltas = price.diff()
    up = deltas.where(deltas > 0, 0)
    down = -deltas.where(deltas < 0, 0)
    up_sum = up.rolling(n).sum()
    down_sum = down.rolling(n).sum()
    rs = up_sum / down_sum
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi
prices['RSI'] = rsi(prices['Close'], 10)

# Larry Williams' R%
def larry_w(prices, period=10):
    hh = prices['High'].rolling(window=period).max()
    ll = prices['Low'].rolling(window=period).min()
    l_w = (hh - prices['Close']) / (hh - ll) * 100
    return l_w
prices['Larry_W'] = larry_w(prices, 10)
prices = prices.dropna()
print(prices)

new_technical_data=prices[['Date','SMA','WMA','Momentum','StochasticK','Stochastic_D','RSI','Larry_W','Close']]
new_technical_data.to_csv('technical_indicators.csv', index=False)