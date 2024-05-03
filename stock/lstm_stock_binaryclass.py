from stock.StockDataBinaryClass import StockDataBinaryClass
from tools.loadFile import loadFile
import numpy as np
import os

dir = r"C:\Users\Never nervius\Desktop\Stock Price 5years"
raw_data1 = loadFile(os.path.join(dir, "META.csv"))
raw_data2 = loadFile(os.path.join(dir, "AMZN.csv"))
raw_data3 = loadFile(os.path.join(dir, "AAPL.csv"))
raw_data4 = loadFile(os.path.join(dir, "GOOG.csv"))
raw_data5 = loadFile(os.path.join(dir, "MSFT.csv"))
raw_data6 = loadFile(os.path.join(dir, "IBM.csv"))
raw_data = np.concatenate((raw_data1, raw_data2, raw_data3, raw_data4, raw_data5, raw_data6), axis=1)

# Available features include ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
features = range(6*6)
prediction = [3]

step = 1  # sample rate
past = 30  # look backward into the past
future = 1  # predict the value in future (1 means right after)
batch_size = 32
GOOG = StockDataBinaryClass(raw_data, step=step, past=past, future=future, batch_size=batch_size,
                            selected_features=features, prediction=prediction,
                            split_rate=0.8)

##############################################################################################
from keras.models import Sequential
from keras import layers
from keras import regularizers
from keras.optimizers import Adam
model = Sequential()

########################################
# model.add(layers.Input(shape=(past, len(features))))
# model.add(layers.LSTM(128, kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
#                       dropout=0.1, recurrent_dropout=0.1, return_sequences=True))
# model.add(layers.LSTM(64, dropout=0.2, recurrent_dropout=0.5))

# #######################################
# model.add(layers.Input(shape=(past, len(features))))
# model.add(layers.Conv1D(128, 7, activation="relu", kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4)))
# model.add(layers.MaxPool1D(pool_size=2, strides=1, padding='same'))
# model.add(layers.Dropout(0.2))
# model.add(layers.BatchNormalization())
# model.add(layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
# model.add(layers.Conv1D(128, 5, activation="relu", kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4)))
# model.add(layers.MaxPool1D(pool_size=2, strides=1, padding='same'))
# model.add(layers.Dropout(0.2))
# model.add(layers.BatchNormalization())
# model.add(layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
# model.add(layers.Conv1D(128, 5, activation="relu", kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4)))
# model.add(layers.MaxPool1D(pool_size=2, strides=1, padding='same'))
# model.add(layers.Dropout(0.2))
# model.add(layers.BatchNormalization())
# model.add(layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
# model.add(layers.Conv1D(64, 3, activation="relu", kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4)))
# model.add(layers.MaxPool1D(pool_size=2, strides=1, padding='same'))
# model.add(layers.Dropout(0.2))
# model.add(layers.BatchNormalization())
# model.add(layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2))


########################################
model.add(layers.Input(shape=(past, len(features), 1)))
model.add(layers.Conv2D(1, [3,3], padding="same", activation="relu", kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4)))
# model.add(layers.MaxPool2D(pool_size=[2,2], strides=[1,1], padding='same'))
# model.add(layers.Dropout(0.2))
model.add(layers.BatchNormalization())
model.add(layers.Reshape((past, -1)))
model.add(layers.LSTM(256, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))

model.add(layers.Reshape((past, -1, 1)))
model.add(layers.Conv2D(1, [3,3], padding="same", activation="relu", kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4)))
# model.add(layers.MaxPool2D(pool_size=[2,2], strides=[1,1], padding='same'))
# model.add(layers.Dropout(0.2))
model.add(layers.BatchNormalization())
model.add(layers.Reshape((past, -1)))
model.add(layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))

model.add(layers.Reshape((past, -1, 1)))
model.add(layers.Conv2D(1, [3,3], padding="same", activation="relu", kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4)))
# model.add(layers.MaxPool2D(pool_size=[2,2], strides=[1,1], padding='same'))
# model.add(layers.Dropout(0.2))
model.add(layers.BatchNormalization())
model.add(layers.Reshape((past, 128)))
model.add(layers.LSTM(256, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))

model.add(layers.Reshape((past, -1, 1)))
model.add(layers.Conv2D(1, [3,3], padding="same", activation="relu", kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4)))
# model.add(layers.MaxPool2D(pool_size=[2,2], strides=[1,1], padding='same'))
# model.add(layers.Dropout(0.2))
model.add(layers.BatchNormalization())
model.add(layers.Reshape((past, -1)))
model.add(layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2))

########################################
model.add(layers.Dense(128))
model.add(layers.Dense(64))
model.add(layers.Dense(25))
model.add(layers.Dense(1, activation='sigmoid'))

learning_rate = 0.0001
# model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# dataset_train, dataset_val = GOOG.get_tfdata()
# history = model.fit(dataset_train, epochs=200, validation_data=dataset_val)

x_train, y_train, x_val, y_val = GOOG.get_npdata()

# # add a dimension for conv2d
# x_train = np.expand_dims(x_train, axis=-1)
# x_val = np.expand_dims(x_val, axis=-1)

history = model.fit(x_train, y_train, epochs=200, validation_data=[x_val, y_val])

# get the real trend distribution
from tools.findTrendCategory import *
print("distribution in y_train:", get_binaryclass_distribution(y_train))
print("distribution in y_val:", get_binaryclass_distribution(y_val))

##############################################################################################
from tools.depict_history import DepictHistory
display = DepictHistory(history)
display.display(title="LSTM Binary Classification")

# # Depict sequences of values
# prediction_train = model.predict(dataset_train)
# prediction_val = model.predict(dataset_val)
# print(prediction_train[-1])
# print(prediction_val[-1])

# from tools import depict_sequence
# data_lists = [[y_train, prediction_train],
#               [y_val, prediction_val]]
# titles = ['real', 'prediction']
# display = depict_sequence.DepictSequence(data_lists, titles)

# # display all sequences
# display.display(title="all sequences")

# # display the last sequence
# display.display(title="the last sequence", start=-past)


