import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

EURUSD = pd.read_csv(r"C:\Users\Never nervius\Desktop\18 years\UPD_EURUSD.csv")
Nasdaq = pd.read_csv(r"C:\Users\Never nervius\Desktop\18 years\NASDAQ.csv")
SP500 = pd.read_csv(r"C:\Users\Never nervius\Desktop\18 years\S&P500.csv")
Amazon = pd.read_csv(r"C:\Users\Never nervius\Desktop\18 years\AMZN.csv")

EURUSD.set_index('Date', inplace=True)
Nasdaq.set_index('Date', inplace=True)
SP500.set_index('Date', inplace=True)
Amazon.set_index('Date', inplace=True)

# Google.drop(columns=['Date'], inplace=True)
# Nasdaq.drop(columns=['Date'], inplace=True)
# SP500.drop(columns=['Date'], inplace=True)
# Amazon.drop(columns=['Date'], inplace=True)

EURUSD.rename(columns={'Close': 'Close'}, inplace=True)
Nasdaq.rename(columns={'Close': 'Nasdaq_Close'}, inplace=True)
SP500.rename(columns={'Close': 'S&P500_Close'}, inplace=True)
Amazon.rename(columns={'Close': 'Amazon_Close'}, inplace=True)
# data = pd.concat([Google, Nasdaq, SP500, Amazon], axis=1)
correlation_matrix = EURUSD.corr()

plt.figure(figsize=(16,8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix of EURUSD")
plt.show()
