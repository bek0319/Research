from stock.StockDataBinaryClass import StockDataBinaryClass
from tools.loadFile import loadFile
from keras.models import Model
import os
import pandas as pd
from keras.models import Sequential
from keras import layers
from keras.optimizers import Adam
from keras import regularizers
from keras.layers import LSTM, Reshape, Conv2D, Input, Flatten, Concatenate, Dense
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
dir = r"C:\Users\Never nervius\Desktop\18 years"
raw_data4 = loadFile(os.path.join(dir, "EURUSD2.csv"))
raw_data4 =raw_data4[2:]
raw_data3 = loadFile(os.path.join(dir, "EURUSD_diff.csv"))
raw_data3 = raw_data3[1:]
raw_data5 = loadFile(os.path.join(dir, "EURUSD_diff2.csv"))
# raw_data3 = raw_data3[1:]
# raw_data = np.concatenate((raw_data4, raw_data1), axis=1)
features1 = range(5)
features2 = range(5)
features3 = range(5)
prediction = [3]

step = 1  # sample rate
past = 60  # look backward into the past
future = 1  # predict the value in future (1 means right after)
batch_size = 128

HIST = StockDataBinaryClass(raw_data4, step=step, past=past, future=future, batch_size=batch_size,
                            selected_features=features1, prediction=prediction,
                            split_rate=0.8)
TI = StockDataBinaryClass(raw_data5, step=step, past=past, future=future, batch_size=batch_size,
                            selected_features=features3, prediction=prediction,
                            split_rate=0.8)
DIFF = StockDataBinaryClass(raw_data3, step=step, past=past, future=future, batch_size=batch_size,
                            selected_features=features2, prediction=prediction,
                            split_rate=0.8)
##############################################################################################
x_train_hist, y_train_hist, x_val_hist, y_val_hist = HIST.get_npdata()
x_train_diff, y_train_diff, x_val_diff, y_val_diff = DIFF.get_npdata()
x_train_ti, y_train_ti, x_val_ti, y_val_ti = TI.get_npdata()
# x_train = x_train[3:]
# x_val = x_val[3:]
learning_rate = 0.0001
x_train_hist = x_train_hist.reshape(x_train_hist.shape[0], 1, past, len(features1))
x_val_hist = x_val_hist.reshape(x_val_hist.shape[0], 1, past, len(features1))
x_train_diff = x_train_diff.reshape(x_train_diff.shape[0], 1, past, len(features2))
x_val_diff = x_val_diff.reshape(x_val_diff.shape[0], 1, past, len(features2))
x_train_ti = x_train_ti.reshape(x_train_ti.shape[0], 1, past, len(features3))
x_val_ti = x_val_ti.reshape(x_val_ti.shape[0], 1, past, len(features3))


# Define input shapes for each channel
input_shape1 = (1, past, len(features1))
input_shape2 = (1, past, len(features2))
input_shape3 = (1, past, len(features3))

# Define input layers for each channel
input_hist = Input(shape=input_shape1)
input_diff = Input(shape=input_shape2)
input_ti = Input(shape=input_shape3)

# First channel - Historical Data
conv1_hist = Conv2D(16, [3,5], data_format="channels_first", padding="same", activation="tanh",
                                   kernel_regularizer=regularizers.L1L2(l1=1e-4, l2=1e-4))(input_hist)
conv2_hist = Conv2D(32, [4, 2], data_format="channels_first", padding="same", activation="tanh",
             kernel_regularizer=regularizers.L1L2(l1=1e-4, l2=1e-4))(conv1_hist)
conv3_hist = Conv2D(64, [6, 2], data_format="channels_first", padding="same", activation="tanh",
             kernel_regularizer=regularizers.L1L2(l1=1e-4, l2=1e-4))(conv2_hist)
conv4_hist = Conv2D(1, [2, 5], data_format="channels_first", padding="same", activation="tanh",
             kernel_regularizer=regularizers.L1L2(l1=1e-4, l2=1e-4))(conv3_hist)

# Second channel - Difference Data
conv1_diff = Conv2D(16, [2, 5], data_format="channels_first", padding="same", activation="tanh",
             kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4))(input_diff)
conv2_diff = Conv2D(64, [4, 3], data_format="channels_first", padding="same", activation="tanh",
                    kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4))(conv1_diff)
conv3_diff = Conv2D(32, [1, 2], data_format="channels_first", padding="same", activation="tanh",
                    kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4))(conv2_diff)
conv4_diff = Conv2D(1, [2, 4], data_format="channels_first", padding="same", activation="tanh",
                    kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4))(conv3_diff)

# Third channel - Technical Indicators
conv1_ti = Conv2D(16, [4, 3], data_format="channels_first", padding="same", activation="tanh",
             kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4))(input_ti)
conv2_ti = Conv2D(64, [4, 2], data_format="channels_first", padding="same", activation="tanh",
             kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4))(conv1_ti)
conv3_ti = Conv2D(32, [1, 5], data_format="channels_first", padding="same", activation="tanh",
             kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4))(conv2_ti)
conv4_ti = Conv2D(1, [2, 2], data_format="channels_first", padding="same", activation="tanh",
             kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4))(conv3_ti)

concatenated = Concatenate(axis=-1)([conv4_hist, conv4_diff, conv4_ti])
conv_after_concat = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(concatenated)
flatten = Flatten()(conv_after_concat)
reshape = Reshape((past, -1))(flatten)
lstm = LSTM(30, dropout=0.2, recurrent_dropout=0.2)(reshape)
output = Dense(1, activation='sigmoid')(lstm)
model = Model(inputs=[input_hist, input_diff, input_ti], outputs=output)

model.compile(optimizer=Adam(learning_rate=learning_rate),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()
# Train the model
history = model.fit([x_train_hist, x_train_diff, x_train_ti], y_train_hist,
                    epochs=100,
                    batch_size=batch_size,
                    validation_data=([x_val_hist, x_val_diff, x_val_ti], y_val_hist))

# ##############################################################################################
from tools.depict_history import DepictHistory

display = DepictHistory(history)
display.display(title="Multiple Channels CNN-LSTM Binary Classification")