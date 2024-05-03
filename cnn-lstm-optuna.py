from stock.StockDataBinaryClass import StockDataBinaryClass
from tools.loadFile import loadFile
import numpy as np
import os
import optuna

dir = r"C:\Users\Never nervius\Desktop\Stock Price 5years"
raw_data1 = loadFile(os.path.join(dir, "META.csv"))
raw_data2 = loadFile(os.path.join(dir, "AMZN.csv"))
raw_data3 = loadFile(os.path.join(dir, "AAPL.csv"))
raw_data4 = loadFile(os.path.join(dir, "GOOG.csv"))
raw_data5 = loadFile(os.path.join(dir, "MSFT.csv"))
raw_data6 = loadFile(os.path.join(dir, "IBM.csv"))
raw_data = np.concatenate((raw_data1, raw_data2, raw_data3, raw_data4, raw_data5, raw_data6), axis=1)

# Available features include ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
features = range(6 * 6)
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
from keras.optimizers import Adam
import tensorflow as tf
from sklearn.metrics import accuracy_score
from keras.layers import Conv2D, MaxPooling2D, Flatten, LSTM, Dense, Dropout


x_train, y_train, x_val, y_val = GOOG.get_npdata()


# Defining the objective function for Optuna
def objective(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    num_layers = trial.suggest_int("num_layers", 2, 6)
    cnn_filters = trial.suggest_int("cnn_filters", 1, 128)
    lstm_units = trial.suggest_int("lstm_units", 16, 128)
    lstm_dropout_rate = trial.suggest_float("lstm_dropout_rate", 0.0, 0.5)
    cnn_l1_reg = trial.suggest_float("cnn_l1_reg", 1e-6, 1e-2, log=True)
    cnn_l2_reg = trial.suggest_float("cnn_l2_reg", 1e-6, 1e-2, log=True)
    num_dense_layers = trial.suggest_int("num_dense_layers", 1, 4)
    num_dense_units = trial.suggest_int("num_dense_units", 16, 256)
    dense_dropout_rate = trial.suggest_float("dense_dropout_rate", 0.0, 0.5)

    model = Sequential()

    for i in range(num_layers):
        if i % 2 == 0:
            if i == 0:
                # Add a CNN layer with regularization
                model.add(layers.Input(shape=(past, len(features), 1)))
                model.add(Conv2D(cnn_filters, (3, 3), activation='relu', padding="same",
                                 kernel_regularizer=tf.keras.regularizers.L1L2(l1=cnn_l1_reg, l2=cnn_l2_reg)))
                model.add(layers.BatchNormalization())
            else:
                model.add(layers.Reshape((past, -1, 1)))
                model.add(Conv2D(cnn_filters, (3, 3), activation='relu', padding="same",
                                 kernel_regularizer=tf.keras.regularizers.L1L2(l1=cnn_l1_reg, l2=cnn_l2_reg)))
                model.add(layers.BatchNormalization())

        else:
            # Add an LSTM layer
            model.add(layers.Reshape((past, -1)))
            model.add(LSTM(lstm_units, return_sequences=True))
            model.add(Dropout(lstm_dropout_rate))

    model.add(tf.keras.layers.Flatten())

    for _ in range(num_dense_layers):
        model.add(Dense(num_dense_units, activation='relu'))
        model.add(Dropout(dense_dropout_rate))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=30, verbose=1)

    # Evaluating the model on the validation set
    y_pred = model.predict(x_val)
    y_pred_binary = (y_pred > 0.5).astype(int)
    accuracy = accuracy_score(y_val, y_pred_binary)

    return accuracy


# Defining optimization parameters
study = optuna.create_study(direction="maximize")  # We want to maximize accuracy
study.optimize(objective, n_trials=15)

# Getting the best hyperparameters
best_params = study.best_params

# Building and training the final model with the best hyperparameters
final_model = Sequential()

for i in range(best_params["num_layers"]):
    if i % 2 == 0:
        if i == 0:
            final_model.add(layers.Input(shape=(past, len(features), 1)))
            final_model.add(Conv2D(best_params["cnn_filters"], (3, 3), activation='relu', padding="same",
                                   kernel_regularizer=tf.keras.regularizers.L1L2(l1=best_params["cnn_l1_reg"],
                                                                                 l2=best_params["cnn_l2_reg"])))
            final_model.add(layers.BatchNormalization())
        else:
            final_model.add(layers.Reshape((past, -1, 1)))
            final_model.add(Conv2D(best_params["cnn_filters"], (3, 3), activation='relu', padding="same",
                                   kernel_regularizer=tf.keras.regularizers.L1L2(l1=best_params["cnn_l1_reg"],
                                                                                 l2=best_params["cnn_l2_reg"])))
            final_model.add(layers.BatchNormalization())

    else:
        final_model.add(layers.Reshape((past, -1)))
        final_model.add(LSTM(best_params["lstm_units"], return_sequences=True))
        final_model.add(Dropout(best_params["lstm_dropout_rate"]))

final_model.add(Flatten())

for _ in range(best_params["num_dense_layers"]):
    final_model.add(Dense(best_params["num_dense_units"], activation='relu'))
    final_model.add(Dropout(best_params["dense_dropout_rate"]))

final_model.add(Dense(1, activation='sigmoid'))

final_model.compile(optimizer=Adam(learning_rate=best_params["learning_rate"]),
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

history = final_model.fit(x_train, y_train, epochs=100, batch_size=32, validation_data=[x_val, y_val])

y_pred = final_model.predict(x_val)
y_pred_binary = (y_pred > 0.5).astype(int)
accuracy_final = accuracy_score(y_val, y_pred_binary)
# print("------------------Y_PRED")
# print(y_pred.shape)
# print("----------------Y_PRED_binary")
# print(y_pred_binary.shape)
# print("y_val------------")
# print(y_val.shape)
# print(x_val.shape)

print(f"Best Hyperparameters: {best_params}")
print(f"Validation Accuracy with Best Model: {accuracy_final}")

# get the real trend distribution
from tools.findTrendCategory import *

print("distribution in y_train:", get_binaryclass_distribution(y_train))
print("distribution in y_val:", get_binaryclass_distribution(y_val))

##############################################################################################
from tools.depict_history import DepictHistory

display = DepictHistory(history)
display.display(title="CNN-LSTM Binary Classification")
