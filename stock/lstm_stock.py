from stock.StockData import StockData
from tools.loadFile import loadFile
import numpy as np

dir = "D:\\Github\\workplace\\20230514_5y\\"
raw_data1 = loadFile(dir + "GOOGL.csv")
raw_data2 = loadFile(dir + "AMZN.csv")
raw_data3 = loadFile(dir + "AAPL.csv")
raw_data4 = loadFile(dir + "META.csv")
raw_data5 = loadFile(dir + "MSFT.csv")
raw_data6 = loadFile(dir + "IBM.csv")
raw_data = np.concatenate((raw_data1, raw_data2, raw_data3, raw_data4, raw_data5, raw_data6), axis=1)

# Available features include ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
features = range(6*6)
prediction = [3]

step = 1  # sample rate
past = 30  # look backward into the past, will be the sequence length
future = 1  # predict the value in the future; 1 means right after, will be the sequence stride
batch_size = 32
GOOG = StockData(raw_data, step=step, past=past, future=future, batch_size=batch_size,
                 selected_features=features, prediction=prediction,
                 split_rate=0.8)

######################################################################################
from keras.models import Sequential
from keras import layers
from keras.optimizers import Adam
from keras import regularizers
model = Sequential()

##############################
# model.add(layers.Input(shape=(past, len(features))))
# model.add(layers.LSTM(128, kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4), 
#     dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
# model.add(layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
# model.add(layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2))

# ##############################
# model.add(layers.Input(shape=(past, len(features))))
# model.add(layers.Conv1D(128, 7, activation="relu", kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4)))
# model.add(layers.BatchNormalization())
# model.add(layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
# model.add(layers.Conv1D(64, 5, activation="relu", kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4)))
# model.add(layers.BatchNormalization())
# model.add(layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
# model.add(layers.Conv1D(64, 5, activation="relu", kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4)))
# model.add(layers.BatchNormalization())
# model.add(layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
# model.add(layers.Conv1D(64, 5, activation="relu", kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4)))
# model.add(layers.BatchNormalization())
# model.add(layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2))

##############################
model.add(layers.Input(shape=(past, len(features), 1)))
model.add(layers.Conv2D(128, [5,5], activation="relu", kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4)))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(64, [5,3], activation="relu", kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4)))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(64, [5,3], activation="relu", kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4)))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(64, [3,3], activation="relu", kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4)))
model.add(layers.BatchNormalization())
# model.add(layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model.add(layers.Flatten())

##############################
model.add(layers.Dense(128))
model.add(layers.Dense(25))
model.add(layers.Dense(1))

# model.compile(optimizer=RMSprop(), loss='mae')
learning_rate = 0.001
model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")
model.summary()

######################################################################################
# dataset_train, dataset_val = GOOG.get_tfdata()
# history = model.fit(dataset_train, epochs=100, validation_data=dataset_val)
# prediction_train = model.predict(dataset_train)
# prediction_val = model.predict(dataset_val)

x_train, y_train, x_val, y_val = GOOG.get_npdata()

# add a dimension for conc2d
x_train = np.expand_dims(x_train, axis=-1)
x_val = np.expand_dims(x_val, axis=-1)

history = model.fit(x_train, y_train, epochs=150, validation_data=[x_val, y_val])
prediction_train = model.predict(x_train)
prediction_val = model.predict(x_val)

# Evaluate the precidtion accuracy in terms of trends, i.e., up or down.
from tools.findTrendCategory import *
prediction_trend = get_trend(prediction_train[1:], y_train[:-1])
real_trend = get_trend(y_train)
acc_trend_train = get_accuracy(prediction_trend, real_trend)
print("accuracy of trend (train):", acc_trend_train)

prediction_trend = get_trend(prediction_val[1:], y_val[:-1])
real_trend = get_trend(y_val)
acc_trend_val = get_accuracy(prediction_trend, real_trend)
print("accuracy of trend (val):", acc_trend_val)

# Get the common-sense predictor: take the last value of a sequence as the prediction
baseline_prediction_train, baseline_prediction_val = GOOG.get_baseline_prediction()
mse_train = (np.square(baseline_prediction_train - y_train)).mean()
mse_val = (np.square(baseline_prediction_val - y_val)).mean()
print("baseline_mse_train:", mse_train)
print("baseline_mse_val:", mse_val)

# Denormalize the data back to original prices
prediction_train = GOOG.denormalize(prediction_train, "train")
prediction_val = GOOG.denormalize(prediction_val, "test")
y_train = GOOG.denormalize(y_train, "train")
y_val = GOOG.denormalize(y_val, "test")
baseline_prediction_train = GOOG.denormalize(baseline_prediction_train, "train")
baseline_prediction_val = GOOG.denormalize(baseline_prediction_val, "test")

# print("train shape:", y_train.shape)
# print(prediction_train.shape)
# print(baseline_prediction_train.shape)

# print("val shape:", y_val.shape)
# print(prediction_val.shape)
# print(baseline_prediction_val.shape)

######################################################################################
# Depict results of training and validation
from tools.depict_allresults import DepictResults
data_lists = [[y_train, prediction_train, baseline_prediction_train],
              [y_val, prediction_val, baseline_prediction_val]]
titles = ['real', 'prediction', 'baseline_prediction']
display = DepictResults(history, data_lists, titles)

s1 = "\nAcc of trend: [" + "{:.2f}".format(acc_trend_train*100) + "%, " + "{:.2f}".format(acc_trend_val*100) + "%]"
s2 = "\nBaseline MSE: [" + "{:.5f}".format(mse_train) + ", " +  "{:.5f}".format(mse_val) + "]"
display.display(title="LSTM-Regression"+s1+s2, show_last_seqeunce=30)



