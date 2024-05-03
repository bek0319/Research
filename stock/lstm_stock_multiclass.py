from stock.StockDataMultiClass import StockDataMultiClass
from tools.loadFile import loadFile
import numpy as np
import os

# Available features include ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
dir = r"C:\Users\Never nervius\Desktop\Stock Price Research"
raw_data1 = loadFile(os.path.join(dir, "META.csv"))
raw_data2 = loadFile(os.path.join(dir, "AMZN.csv"))
raw_data3 = loadFile(os.path.join(dir, "AAPL.csv"))
raw_data4 = loadFile(os.path.join(dir, "GOOG.csv"))
raw_data5 = loadFile(os.path.join(dir, "MSFT.csv"))
raw_data6 = loadFile(os.path.join(dir, "IBM.csv"))

# raw_data1 = loadFile(dir + "\GOOG.csv")
# raw_data2 = loadFile(dir + "\AMZN.csv")
# raw_data3 = loadFile(dir + "\AAPL.csv")
# raw_data4 = loadFile(dir + "\META.csv")
# raw_data5 = loadFile(dir + "\MSFT.csv")
# raw_data6 = loadFile(dir + "\IBM.csv")
raw_data = np.concatenate((raw_data1, raw_data2, raw_data3, raw_data4, raw_data5, raw_data6), axis=1)

features = range(6*6)
prediction = [3]

step = 1  # sample rate
past = 30  # look backward into the past
future = 1  # predict the value in future (1 means right after)
batch_size = 32
GOOG = StockDataMultiClass(raw_data, step=step, past=past, future=future, batch_size=batch_size,
                           selected_features=features, prediction=prediction,
                           split_rate=0.8)
# GOOG.set_class_division([-0.05, 0.05])
GOOG.set_class_division([-0.03, 0.0, 0.03])

##############################################################################################
from keras.models import Sequential
from keras import layers
from keras import regularizers
from keras.optimizers import Adam
model = Sequential()

# ######################################
# model.add(layers.Input(shape=(past, len(features))))
# model.add(layers.LSTM(128, kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
#                       dropout=0.1, recurrent_dropout=0.1, return_sequences=True))
# model.add(layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
# model.add(layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2))

# ######################################
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
# model.add(layers.Conv1D(64, 3, activation="relu", kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4)))
# model.add(layers.MaxPool1D(pool_size=2, strides=1, padding='same'))
# model.add(layers.Dropout(0.2))
# model.add(layers.BatchNormalization())
# model.add(layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
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

# ########################################
# model.add(layers.Input(shape=(past, len(features), 1)))
# model.add(layers.ConvLSTM1D(past, 3, padding="same", activation="relu", kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
#                             dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
# model.add(layers.ConvLSTM1D(past, 3, padding="same", activation="relu", kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
#                             dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
# model.add(layers.ConvLSTM1D(past, 3, padding="same", activation="relu", kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
#                             dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
# model.add(layers.ConvLSTM1D(past, 3, padding="same", activation="relu", kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
#                             dropout=0.2, recurrent_dropout=0.2))
# model.add(layers.Flatten())

########################################
model.add(layers.Dense(128))
model.add(layers.Dense(64))
model.add(layers.Dense(25))
model.add(layers.Dense(GOOG.get_number_classes(), activation='softmax'))

learning_rate = 0.0001
# model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# dataset_train, dataset_val = GOOG.get_tfdata()
# history = model.fit(dataset_train, epochs=200, validation_data=dataset_val)

x_train, y_train, x_val, y_val = GOOG.get_npdata_reordered()

# add a dimension for conv2d
# x_train = np.expand_dims(x_train, axis=-1)
# x_val = np.expand_dims(x_val, axis=-1)
print("x_train reshape:", x_train.shape)
print("x_val reshape:", x_val.shape)

history = model.fit(x_train, y_train, epochs=200, validation_data=[x_val, y_val])

# get the real trend distribution
from tools.findTrendCategory import *
print("distribution in y_train:", get_multiclass_distribution(y_train))
print("distribution in y_val:", get_multiclass_distribution(y_val))

##############################################################################################
from tools.depict_history import DepictHistory
display = DepictHistory(history)
display.display(title="LSTM Multi Classification")
