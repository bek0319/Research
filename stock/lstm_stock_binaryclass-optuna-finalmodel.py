from stock.StockDataBinaryClass import StockDataBinaryClass
from tools.loadFile import loadFile
import numpy as np
import os
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras import layers
from keras.optimizers import Adam
from keras import regularizers
from keras.layers import LSTM, Bidirectional, Reshape, Conv2D, Input, Flatten, Dense
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
dir = r"C:\Users\Never nervius\Desktop\18 years"
# raw_data3 = loadFile(os.path.join(dir, "UPD_EURGBP_test.csv"))
raw_data4 = loadFile(os.path.join(dir, "EURUSD2.csv"))
# raw_data4 = raw_data4[29:]
raw_data3 = loadFile(os.path.join(dir, "EURUSD2.csv"))
# raw_data3 = raw_data3[1:]
# # raw_data5 = loadFile(os.path.join(dir, "MSFT.csv"))
# # raw_data6 = loadFile(os.path.join(dir, "IBM.csv"))
# raw_data = np.concatenate((raw_data4, raw_data1), axis=1)
# # Available features include ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']


features1 = range(5)
features2 = range(5)
prediction = [3]

step = 1  # sample rate
past = 60  # look backward into the past
future = 1  # predict the value in future (1 means right after)
batch_size = 128

EURUSD1 = StockDataBinaryClass(raw_data3, step=step, past=past, future=future, batch_size=batch_size,
                            selected_features=features2, prediction=prediction,
                            split_rate=0.8)
EURUSD2 = StockDataBinaryClass(raw_data4, step=step, past=past, future=future, batch_size=batch_size,
                            selected_features=features1, prediction=prediction,
                            split_rate=0.8)
##############################################################################################
x_train, y_train, x_val, y_val = EURUSD1.get_npdata()
x_train1, y_train1, x_val1, y_val1 = EURUSD2.get_npdata()
x_train = x_train[2:]
x_val = x_val[2:]
print(y_train1[:15])
# learning_rate = 0.0001
# x_train = x_train.reshape(x_train.shape[0],1,  past, len(features2))
# x_val = x_val.reshape(x_val.shape[0],1,  past, len(features2))
# # x_val1 = x_val1.reshape(x_val1.shape[0],1,  past, len(features))
#
#
#
# model = Sequential()
# model.add(Input(shape=(1, past, len(features2))))
# model.add(Conv2D(16, [2,5], data_format="channels_first", padding="same", activation="tanh",
#                                    kernel_regularizer=regularizers.L1L2(l1=1e-4, l2=1e-4)))
# model.add(Conv2D(32, [5, 4], data_format="channels_first", padding="same", activation="tanh",
#              kernel_regularizer=regularizers.L1L2(l1=1e-4, l2=1e-4))),
# model.add(Conv2D(128, [3, 2], data_format="channels_first", padding="same", activation="tanh",
#              kernel_regularizer=regularizers.L1L2(l1=1e-4, l2=1e-4))),
# model.add(Conv2D(64, [4, 3], data_format="channels_first", padding="same", activation="tanh",
#              kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4))),
# model.add(Conv2D(32, [4, 2], data_format="channels_first", padding="same", activation="tanh",
#              kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4))),
# model.add(Conv2D(1, [2, 4], data_format="channels_first", padding="same", activation="tanh",
#              kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4))),
# model.add(Reshape((past, len(features2))))
# model.add(LSTM(30, dropout=0.2, recurrent_dropout=0.2))
# model.add(layers.Dense(1, activation='sigmoid'))
#
# model.compile(optimizer=Adam(learning_rate=learning_rate),
#               loss='binary_crossentropy',
#               metrics=['accuracy'])
#
# model.summary()
#
#
# history = model.fit(x_train, y_train1, epochs=100, validation_data=(x_val, y_val1))
#
# # Plotting Accuracy and Loss
# plt.figure(figsize=(10, 6))
#
# # Plotting Loss with dots
# plt.plot(history.history['loss'], label='Training Loss', color='blue', marker='o', linestyle='-')
# plt.plot(history.history['val_loss'], label='Validation Loss', color='orange', marker='o', linestyle='-')
# plt.xlabel('Epochs')
# plt.ylabel('Loss', color='blue')
# plt.tick_params(axis='y', labelcolor='blue')
# plt.legend(loc='upper left')
#
# # Creating a twin y-axis to plot accuracy
# ax2 = plt.gca().twinx()
# ax2.plot(history.history['accuracy'], label='Training Accuracy', color='green', marker='o', linestyle='-')
# ax2.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red', marker='o', linestyle='-')
# ax2.set_ylabel('Accuracy', color='green')
# ax2.tick_params(axis='y', labelcolor='green')
# ax2.legend(loc='upper right')
#
# plt.title('Training Loss and Accuracy')
# plt.xlabel('Epochs')
#
# # Adding grid
# plt.grid(True, linestyle='--', alpha=0.7)
#
# # Tight layout to prevent clipping of labels
# plt.tight_layout()
#
# plt.show()
#
#
#








# ##############################################################################################
# from tools.depict_history import DepictHistory
#
# display = DepictHistory(history)
# display.display(title="CNN-LSTM Binary Classification")






# # Evaluation on the new dataset
# evaluation_metrics = model.evaluate(x_val1, y_val1)
# print("Evaluation Loss:", evaluation_metrics[0])
# print("Evaluation Accuracy:", evaluation_metrics[1])
#
# # Make predictions on the new dataset
# predictions = model.predict(x_val1)
#
# # Compare predictions with ground truth labels
# predicted_labels = (predictions > 0.5).astype(int)
#
# # Evaluate performance metrics
# accuracy = accuracy_score(y_val1, predicted_labels)
# precision = precision_score(y_val1, predicted_labels)
# recall = recall_score(y_val1, predicted_labels)
# f1 = f1_score(y_val1, predicted_labels)
#
# print("Accuracy of the last 60days:", accuracy)
# print("Precision of the last 60days:", precision)
# print("Recall of the last 60days:", recall)
# print("F1 Score of the last 60days:", f1)
#
# # get the real trend distribution
# from tools.findTrendCategory import *
#
# print("distribution in y_train:", get_binaryclass_distribution(y_train))
# print("distribution in y_val:", get_binaryclass_distribution(y_val))
#
# # Visualize training history
# display = DepictHistory(history)
# display.display(title="CNN-LSTM Binary Classification")