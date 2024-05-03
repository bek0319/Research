from stock.StockDataBinaryClass import StockDataBinaryClass
from tools.loadFile import loadFile
import numpy as np
import os
import optuna
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

dir = r"C:\Users\Never nervius\Desktop\18 years"
raw_data1 = loadFile(os.path.join(dir, "NASDAQ.csv"))
raw_data2 = loadFile(os.path.join(dir, "GOOG.csv"))
raw_data = np.concatenate((raw_data1, raw_data2), axis=1)

# Available features include ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
features = range(6 * 2)
prediction = [1]

step = 1  # sample rate
past = 60  # look backward into the past
future = 1  # predict the value in future (1 means right after)
batch_size = 64
GOOG = StockDataBinaryClass(raw_data, step=step, past=past, future=future, batch_size=batch_size,
                            selected_features=features, prediction=prediction,
                            split_rate=0.8)

##############################################################################################
from keras.models import Sequential
from keras import layers
from keras.optimizers import Adam
import tensorflow as tf

x_train, y_train, x_val, y_val = GOOG.get_npdata()


# Defining the objective function for Optuna
def objective(trial):
    learning_rate = trial.suggest_float("learning_rate", 0.00001, 0.1)
    cnn_filters1 = trial.suggest_int("cnn_filters1", 1, 128)
    cnn_filters2 = trial.suggest_int("cnn_filters2", 1, 128)
    cnn_filters3 = trial.suggest_int("cnn_filters3", 1, 128)
    cnn_filters4 = trial.suggest_int("cnn_filters4", 1, 128)
    lstm_units1 = trial.suggest_int("lstm_units1", 1, 256)
    lstm_units2 = trial.suggest_int("lstm_units2", 1, 256)
    lstm_dropout_rate = trial.suggest_float("lstm_dropout_rate", 0.0, 0.5)
    cnn_l1_reg = trial.suggest_float("cnn_l1_reg", 1e-6, 1e-2, log=True)
    cnn_l2_reg = trial.suggest_float("cnn_l2_reg", 1e-6, 1e-2, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
    dense_units1 = trial.suggest_int("dense_units1", 1, 128)
    dense_units2 = trial.suggest_int("dense_units2", 1, 128)

    model = Sequential()
    model.add(layers.Input(shape=(past, len(features), 1)))
    model.add(layers.Conv2D(cnn_filters1, [3, 3], padding="same", activation="relu",
                            kernel_regularizer=tf.keras.regularizers.L1L2(l1=cnn_l1_reg, l2=cnn_l2_reg)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(cnn_filters2, [3, 3], padding="same", activation="relu",
                            kernel_regularizer=tf.keras.regularizers.L1L2(l1=cnn_l1_reg, l2=cnn_l2_reg)))
    model.add(layers.MaxPool2D(pool_size=[2, 2], strides=[1, 1], padding='same'))
    # model.add(layers.Dropout(0.2))
    model.add(layers.BatchNormalization())
    model.add(layers.Reshape((past, -1)))
    model.add(layers.LSTM(lstm_units1, dropout=dropout_rate, recurrent_dropout=lstm_dropout_rate, return_sequences=True))

    #Second CNN-CNN-LSTM
    model.add(layers.Reshape((past, -1, 1)))
    model.add(layers.Conv2D(cnn_filters3, [3, 3], padding="same", activation="relu",
                            kernel_regularizer=tf.keras.regularizers.L1L2(l1=cnn_l1_reg, l2=cnn_l2_reg)))
    model.add(layers.BatchNormalization())
    model.add(layers.Reshape((past, -1, 1)))
    model.add(layers.Conv2D(cnn_filters4, [3, 3], padding="same", activation="relu",
                            kernel_regularizer=tf.keras.regularizers.L1L2(l1=cnn_l1_reg, l2=cnn_l2_reg)))
    model.add(layers.MaxPool2D(pool_size=[2, 2], strides=[1, 1], padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Reshape((past, -1)))
    model.add(layers.LSTM(lstm_units2, dropout=dropout_rate, recurrent_dropout=lstm_dropout_rate, return_sequences=True))

    model.add(layers.Dense(dense_units1))
    model.add(layers.Dense(dense_units2))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=20, verbose=1, validation_data=[x_val, y_val])
    loss, accuracy = model.evaluate(x_val, y_val, verbose=0)

    return accuracy


# Defining optimization parameters
study = optuna.create_study(direction="maximize")  # We want to maximize accuracy
study.optimize(objective, n_trials=30)

# Getting the best hyperparameters
best_params = study.best_params

# Building and training the final model with the best hyperparameters
final_model = Sequential()
final_model.add(layers.Input(shape=(past, len(features), 1)))
final_model.add(layers.Conv2D(best_params['cnn_filters1'], [3, 3], padding="same", activation="relu",
                        kernel_regularizer=tf.keras.regularizers.L1L2(l1=best_params['cnn_l1_reg'], l2=best_params['cnn_l2_reg'])))
final_model.add(layers.BatchNormalization())
final_model.add(layers.Conv2D(best_params['cnn_filters2'], [3, 3], padding="same", activation="relu",
                        kernel_regularizer=tf.keras.regularizers.L1L2(l1=best_params['cnn_l1_reg'], l2=best_params['cnn_l2_reg'])))
final_model.add(layers.MaxPool2D(pool_size=[2, 2], strides=[1, 1], padding='same'))
# model.add(layers.Dropout(0.2))
final_model.add(layers.BatchNormalization())
final_model.add(layers.Reshape((past, -1)))
final_model.add(layers.LSTM(best_params['lstm_units1'], dropout=best_params['dropout_rate'], recurrent_dropout=best_params['lstm_dropout_rate'], return_sequences=True))

# Second CNN-CNN-LSTM
final_model.add(layers.Reshape((past, -1, 1)))
final_model.add(layers.Conv2D(best_params['cnn_filters3'], [3, 3], padding="same", activation="relu",
                        kernel_regularizer=tf.keras.regularizers.L1L2(l1=best_params['cnn_l1_reg'], l2=best_params['cnn_l2_reg'])))
final_model.add(layers.BatchNormalization())
final_model.add(layers.Reshape((past, -1, 1)))
final_model.add(layers.Conv2D(best_params['cnn_filters4'], [3, 3], padding="same", activation="relu",
                        kernel_regularizer=tf.keras.regularizers.L1L2(l1=best_params['cnn_l1_reg'], l2=best_params['cnn_l2_reg'])))
final_model.add(layers.MaxPool2D(pool_size=[2, 2], strides=[1, 1], padding='same'))
final_model.add(layers.BatchNormalization())
final_model.add(layers.Reshape((past, -1)))
final_model.add(layers.LSTM(best_params['lstm_units2'], dropout=best_params['dropout_rate'], recurrent_dropout=best_params['lstm_dropout_rate'], return_sequences=True))

final_model.add(layers.Dense(best_params['dense_units1']))
final_model.add(layers.Dense(best_params['dense_units2']))
final_model.add(layers.Dense(1, activation='sigmoid'))
final_model.compile(optimizer=Adam(learning_rate=best_params['learning_rate']),
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = final_model.fit(x_train, y_train, epochs=100, batch_size=64, validation_data=[x_val, y_val])

_, accuracy_final = final_model.evaluate(x_val, y_val)

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
