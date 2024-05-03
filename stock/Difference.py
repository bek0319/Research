import pandas as pd

data = pd.read_csv(r"C:\Users\Never nervius\Desktop\18 years\EURUSD2.csv")

data1 = data.drop('Date', axis=1).apply(pd.to_numeric, errors='coerce')
data_3_days_later = data1.shift(-3)
difference_3_days_later = data_3_days_later - data1
dates = data['Date'][:-3]
data_diff_3_days_later = pd.concat([dates.reset_index(drop=True), difference_3_days_later.iloc[:-3]], axis=1)

file_path = r"C:\Users\Never nervius\Desktop\18 years\EURUSD_diff3.csv"
data_diff_3_days_later.to_csv(file_path, index=False)