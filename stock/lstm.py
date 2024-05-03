from stock.StockDataBinaryClass import StockDataBinaryClass
from tools.loadFile import loadFile
import numpy as np
import os
import optuna
import keras
import tensorflow as tf
from sklearn.metrics import accuracy_score

dir = r"C:\Users\Never nervius\Desktop\Stock Price 5years"
raw_data1 = loadFile(os.path.join(dir, "META.csv"))
raw_data2 = loadFile(os.path.join(dir, "AMZN.csv"))
raw_data3 = loadFile(os.path.join(dir, "AAPL.csv"))
raw_data4 = loadFile(os.path.join(dir, "GOOG.csv"))
raw_data5 = loadFile(os.path.join(dir, "MSFT.csv"))
raw_data6 = loadFile(os.path.join(dir, "IBM.csv"))
raw_data = np.concatenate((raw_data1, raw_data2, raw_data3, raw_data4, raw_data5, raw_data6),
                          axis=1)

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

x_train, y_train, x_val, y_val = GOOG.get_npdata()


def objective(trial):
    num_units = trial.suggest_int("num_units", 16, 256)
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    num_lstm_layers = trial.suggest_int("num_lstm_layers", 1, 4)  # Hyperparameter for LSTM layers
    num_dense_layers = trial.suggest_int("num_dense_layers", 1, 4)  # Hyperparameter for Dense layers

    model = tf.keras.Sequential()

    # Add LSTM layers based on the suggested number of layers
    for _ in range(num_lstm_layers):
        model.add(tf.keras.layers.LSTM(num_units, activation='relu', return_sequences=True))
        model.add(tf.keras.layers.Dropout(dropout_rate))

    # Flatten or reshape output for Dense layers
    model.add(tf.keras.layers.Flatten())  # You can use Flatten or Reshape based on your needs

    # Add Dense layers based on the suggested number of layers
    for _ in range(num_dense_layers):
        model.add(tf.keras.layers.Dense(num_units, activation='relu'))
        model.add(tf.keras.layers.Dropout(dropout_rate))

    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=1, batch_size=32, verbose=1)

    # Evaluate the model on the validation set
    y_pred = model.predict(x_val)
    y_pred_binary = (y_pred > 0.5).astype(int)
    accuracy = accuracy_score(y_val, y_pred_binary)

    return accuracy  # Optimize for accuracy (negative value to minimize)


# Create an Optuna study and optimize hyperparameters
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=1)

# Get the best hyperparameters
best_params = study.best_params

# Build and train the final model with the best hyperparameters
final_model = tf.keras.Sequential()

for _ in range(best_params["num_lstm_layers"]):
    final_model.add(tf.keras.layers.LSTM(best_params["num_units"], activation='relu', return_sequences=True))
    final_model.add(tf.keras.layers.Dropout(best_params["dropout_rate"]))

final_model.add(tf.keras.layers.Flatten())

for _ in range(best_params["num_dense_layers"]):
    final_model.add(tf.keras.layers.Dense(best_params["num_units"], activation='relu'))
    final_model.add(tf.keras.layers.Dropout(best_params["dropout_rate"]))

final_model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

final_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=best_params["learning_rate"]),
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

history = final_model.fit(x_train, y_train, epochs=1, batch_size=32, validation_data=[x_val, y_val])

# Evaluate the final model
y_pred = final_model.predict(x_val)
y_pred_binary = (y_pred > 0.5).astype(int)
accuracy = accuracy_score(y_val, y_pred_binary)
print("--------y_pred numbers:")
print(y_pred.shape)
print("----------------y_pred_binary numbers::::")
print(y_pred_binary.shape)
print("y_val-------- -  ")
print(y_val.shape)


print(f"Best Hyperparameters: {best_params}")
print(f"Validation Accuracy with Best Model: {accuracy}")

# get the real trend distribution
from tools.findTrendCategory import *

print("distribution in y_train:", get_binaryclass_distribution(y_train))
print("distribution in y_val:", get_binaryclass_distribution(y_val))

##############################################################################################
from tools.depict_history import DepictHistory

display = DepictHistory(history)
display.display(title="LSTM Binary Classification")
