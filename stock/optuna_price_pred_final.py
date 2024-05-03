from stock.StockDataBinaryClass import StockDataBinaryClass
from tools.loadFile import loadFile
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
dir = r"C:\Users\Never nervius\Desktop\18 years"
# raw_data1 = loadFile(os.path.join(dir, "Gold.csv"))
raw_data4 = loadFile(os.path.join(dir, "NASDAQ.csv"))
# # raw_data5 = loadFile(os.path.join(dir, "MSFT.csv"))
# # raw_data6 = loadFile(os.path.join(dir, "IBM.csv"))
# raw_data = np.concatenate((raw_data4, raw_data3), axis=1)
# # Available features include ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
features = range(6)
prediction = [3]

step = 1  # sample rate
past = 60  # look backward into the past
future = 1  # predict the value in future (1 means right after)
batch_size = 64
GOOG = StockDataBinaryClass(raw_data4, step=step, past=past, future=future, batch_size=batch_size,
                            selected_features=features, prediction=prediction,
                            split_rate=0.7)

##############################################################################################
from keras.models import Sequential
from keras import layers
from keras.optimizers import Adam
from keras import regularizers
from keras.layers import LSTM, Bidirectional, Reshape, Conv2D, BatchNormalization, MaxPool2D, Input, Flatten, Dense, MaxPooling2D

x_train, y_train, x_val, y_val = GOOG.get_number()

# print(x_val[:1])
# print(y_train.shape)
# print(x_val.shape)
# print(y_val.shape)

#hyperparameters
learning_rate = 0.06
lstm_units1 = 100
lstm_units2 = 100
# lstm_units3 = 50
# lstm_units4 = 200
cnn_filters1 = 20
# cnn_filters2 = 64
# cnn_filters3 = 128
# cnn_filters4 = 128
# cnn_filters5 = 64
# cnn_filters6 = 128
lstm_dropout_rate = 0.03
cnn_l1_reg = 0.04
cnn_l2_reg = 0.09
dropout_rate = 0.003


model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(past, len(features), 1), padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization())
model.add(Reshape((past, -1)))  # Reshape to (60, -1)
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1))

# Compile the model
optimizer = Adam(learning_rate=0.001)  # Adjust the learning rate if needed
model.compile(optimizer=optimizer, loss='mse')

# Train the model
history = model.fit(x_train, y_train, epochs=50, batch_size=64, validation_data=(x_val, y_val))

# Make predictions
predictions_val = model.predict(x_val)
print("Here is the predictions:")
print(predictions_val)
# # Plotting
validation = {}
validation['y_val'] = y_val
validation['Predictions'] = predictions_val
plt.figure(figsize=(16, 8), dpi=100)
plt.title('CNN-LSTM')
plt.xlabel('Days')
plt.ylabel('Price')
plt.plot(y_train, label='Training', color='blue')

# Plotting true values and predictions with a gap of 60 days
gap = 60
train_end_index = len(y_train)
validation_start_index = train_end_index + gap

# Plotting true values and predictions with proper x-axis indices
plt.plot(np.arange(train_end_index, train_end_index + len(validation['y_val'])), validation['y_val'], label='True Values', color='green')
plt.plot(np.arange(train_end_index, train_end_index + len(validation['Predictions'])), validation['Predictions'], label='Predictions', color='red')

plt.legend(loc='lower right')

# Set x-axis range to include both training and validation data
plt.xlim(0, len(y_train) + len(validation['y_val']) - 1)

plt.show()
