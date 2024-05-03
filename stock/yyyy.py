import pandas as pd
import pandas_ta as ta
import numpy as np

# Load your dataset
# data = pd.read_csv(r"C:\Users\Never nervius\Desktop\18 years\EURUSD.csv")
data = pd.read_csv(r"C:\Users\Never nervius\Desktop\18 years\EURUSD2.csv")

# Calculate the differences between consecutive rows for all columns except 'Date'
diff_df = data.drop('Date', axis=1).diff().dropna()

# Extract the dates from the original DataFrame, excluding the first date
dates = data['Date'][1:]

# Combine dates with differences
data_diff = pd.concat([dates.reset_index(drop=True), diff_df.reset_index(drop=True)], axis=1)


#
# # Calculate additional technical indicators using pandas_ta
# data['EMA_20'] = ta.ema(data['Close'], length=20)  # Exponential Moving Average (20)
# data['WMA_20'] = ta.wma(data['Close'], length=20)  # Weighted Moving Average (20)
# data['ATR_14'] = ta.atr(data['High'], data['Low'], data['Close'], length=14)  # Average True Range (14)
# data['RSI_14'] = ta.rsi(data['Close'], length=14)  # Relative Strength Index (14)
# stoch = ta.stoch(data['High'], data['Low'], data['Close'])
# data['Stoch_%K'] = stoch['STOCHk_14_3_3']  # Stochastic %K
# data['Stoch_%D'] = stoch['STOCHd_14_3_3']  # Stochastic %D
# data['ROC_20'] = ta.roc(data['Close'], length=20)  # Rate of Change (20)
# adx_data = ta.adx(data['High'], data['Low'], data['Close'], length=14)  # Average Directional Index (14)
# data['ADX_14'] = adx_data['ADX_14']  # Assigning only ADX to 'ADX_14' column
# data['CCI'] = ta.cci(data['High'], data['Low'], data['Close'])  # Commodity Channel Index (CCI)
# # data['SAR'] = ta.SAR(data['High'], data['Low'])  # Parabolic Stop and Reverse (SAR)
# # data['DEMA_20'] = ta.dema(data['Close'], length=20)  # Double Exponential Moving Average (20)
# # data['ROC_14'] = ta.roc(data['Close'], length=14)  # Rate of Change (14)
# # data['T3_20'] = ta.t3(data['Close'], length=20)  # T3 Moving Average (20)
# # Calculate RS (Relative Strength)
# delta = data['Close'].diff()
# gain = delta.mask(delta < 0, 0).rolling(window=14).mean()
# loss = (-delta).mask(delta > 0, 0).rolling(window=14).mean()
# rs = gain / loss
# data['RS'] = rs  # RS = AvgU/AvgD
# max_window = 30
# data = data[max_window - 1:]
#
# # Save the updated dataset with new technical indicators
file_path = r"C:\Users\Never nervius\Desktop\18 years\EURUSD_diff.csv"
data_diff.to_csv(file_path, index=False)