import pandas as pd

stock1_data = pd.read_csv(r"C:\Users\Never nervius\Desktop\18 years\UPD_EURUSD2.csv")
# print(stock1_data.shape)
# data1 = pd.read_csv(r"C:\Users\Never nervius\Desktop\18 years\EURGBP.csv")
data1 = stock1_data.drop(['Close', 'Adj Close', 'High', 'Low', 'Open'], axis=1)
data2 = data1.dropna()
#
# # Load datasets into pandas dataframes
# print(data1.shape)
# print(stock2_data.shape)
# # Convert 'date' column to datetime type for both dataframes
# stock1_data['Date'] = pd.to_datetime(stock1_data['Date'])
# stock2_data['Date'] = pd.to_datetime(stock2_data['Date'])
#
#
# indices_to_delete = [181, 1647]
#
# # Drop rows with these indices from both datasets
# # stock1_data = stock1_data.drop(indices_to_delete)
# stock2_data = stock2_data.drop(indices_to_delete)
#
# # Optionally, you can reset the index of the DataFrame
# # stock1_data.reset_index(drop=True, inplace=True)
# stock2_data.reset_index(drop=True, inplace=True)
# print(stock1_data)
# print(stock2_data)
#
# # Identify dates in stock1_data that are not in stock2_data
# missing_dates_stock1 = stock1_data[~stock1_data['Date'].isin(stock2_data['Date'])]
#
# # Identify dates in stock2_data that are not in stock1_data
# missing_dates_stock2 = stock2_data[~stock2_data['Date'].isin(stock1_data['Date'])]
#
# # Display dates in stock1_data not present in stock2_data
# print("Dates in stock1_data not present in stock2_data:")
# print(missing_dates_stock1['Date'])
#
# # Display dates in stock2_data not present in stock1_data
# print("\nDates in stock2_data not present in stock1_data:")
# print(missing_dates_stock2['Date'])

file_path = r"C:\Users\Never nervius\Desktop\18 years\UPD_EURUSD5.csv"
data2.to_csv(file_path, index=False)