from stock.StockDataBinaryClass import StockDataBinaryClass
from tools.loadFile import loadFile
import numpy as np
import os
import optuna
import matplotlib.pyplot as plt
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

dir = r"C:\Users\Never nervius\Desktop\18 years"
raw_data4 = loadFile(os.path.join(dir, "EURUSD2.csv"))
raw_data4 = raw_data4[1:]
raw_data3 = loadFile(os.path.join(dir, "EURUSD_diff.csv"))
features = range(5)
prediction = [3]

step = 1  # sample rate
past = 60  # look backward into the past
future = 1  # predict the value in future (1 means right after)
batch_size = 128

EURUSD1 = StockDataBinaryClass(raw_data3, step=step, past=past, future=future, batch_size=batch_size,
                            selected_features=features, prediction=prediction,
                            split_rate=0.8)
EURUSD2 = StockDataBinaryClass(raw_data4, step=step, past=past, future=future, batch_size=batch_size,
                            selected_features=features, prediction=prediction,
                            split_rate=0.8)

##############################################################################################
from keras.models import Sequential
from keras import layers
from keras.optimizers import Adam
import tensorflow as tf
from sklearn.metrics import accuracy_score
from keras.layers import Input,  Conv2D, LSTM, Dense, Dropout, Reshape

x_train, y_train, x_val, y_val = EURUSD1.get_npdata()
x_train1, y_train1, x_val1, y_val1 = EURUSD2.get_npdata()
x_train = x_train.reshape(x_train.shape[0], past, len(features), 1)
x_val = x_val.reshape(x_val.shape[0], past, len(features), 1)
x_val1 = x_val1.reshape(x_val1.shape[0], past, len(features), 1)
y_train1 = np.expand_dims(y_train1, axis=1)  # Add an extra dimension to match the shape of logits
y_val1 = np.expand_dims(y_val1, axis=1)  # Add an extra dimension to match the shape of logits
# Defining the objective function for Optuna
def objective(trial):
    learning_rate = trial.suggest_float("learning_rate", 0.0001, 0.1)
    cnn_filters1 = trial.suggest_int("cnn_filters1", 1, 256)
    cnn_filters2 = trial.suggest_int("cnn_filters2", 1, 256)
    cnn_filters3 = trial.suggest_int("cnn_filters3", 1, 256)
    cnn_filters4 = trial.suggest_int("cnn_filters4", 1, 256)
    # lstm_units1 = trial.suggest_int("lstm_units1", 1, 256)
    # lstm_dropout_rate = trial.suggest_float("lstm_dropout_rate", 0.0, 0.06)
    cnn_l1_reg = trial.suggest_float("cnn_l1_reg", 1e-6, 1e-2, log=True)
    cnn_l2_reg = trial.suggest_float("cnn_l2_reg", 1e-6, 1e-2, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.06)

    model = Sequential()
    ########################################
    model.add(Input(shape=(past, len(features), 1)))
    model.add(Conv2D(cnn_filters1, [2, 5], data_format="channels_first", padding="same", activation="tanh",
                            kernel_regularizer=tf.keras.regularizers.L1L2(l1=cnn_l1_reg, l2=cnn_l2_reg)))
    model.add(Conv2D(cnn_filters2, [5, 4], data_format="channels_first", padding="same", activation="tanh",
                            kernel_regularizer=tf.keras.regularizers.L1L2(l1=cnn_l1_reg, l2=cnn_l2_reg)))
    model.add(Conv2D(cnn_filters3, [3, 2], data_format="channels_first", padding="same", activation="tanh",
                            kernel_regularizer=tf.keras.regularizers.L1L2(l1=cnn_l1_reg, l2=cnn_l2_reg)))
    model.add(Conv2D(cnn_filters4, [4, 3], data_format="channels_first", padding="same", activation="tanh",
                            kernel_regularizer=tf.keras.regularizers.L1L2(l1=cnn_l1_reg, l2=cnn_l2_reg)))

    ########################################
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train1, epochs=40, verbose=1, validation_data=(x_val, y_val1))
    loss, accuracy = model.evaluate(x_val, y_val1, verbose=0)
    return accuracy


# Defining optimization parameters
study = optuna.create_study(direction="maximize")  # We want to maximize accuracy
study.optimize(objective, n_trials=50)

# Getting the best hyperparameters
best_params = study.best_params

# Building and training the final model with the best hyperparameters
final_model = Sequential()

########################################
########################################
final_model.add(Input(shape=(past, len(features), 1)))
final_model.add(Conv2D(best_params['cnn_filters1'], [2, 5],data_format="channels_first", padding="same", activation="tanh",
                 kernel_regularizer=tf.keras.regularizers.L1L2(l1=best_params['cnn_l1_reg'], l2=best_params['cnn_l2_reg'])))
final_model.add(Conv2D(best_params['cnn_filters2'], [5, 4],data_format="channels_first", padding="same", activation="tanh",
                 kernel_regularizer=tf.keras.regularizers.L1L2(l1=best_params['cnn_l1_reg'], l2=best_params['cnn_l2_reg'])))
final_model.add(Conv2D(best_params['cnn_filters3'], [3, 2],data_format="channels_first", padding="same", activation="tanh",
                 kernel_regularizer=tf.keras.regularizers.L1L2(l1=best_params['cnn_l1_reg'], l2=best_params['cnn_l2_reg'])))
final_model.add(Conv2D(best_params['cnn_filters4'], [4, 3],data_format="channels_first", padding="same", activation="tanh",
                 kernel_regularizer=tf.keras.regularizers.L1L2(l1=best_params['cnn_l1_reg'], l2=best_params['cnn_l2_reg'])))

########################################
final_model.add(Dense(1, activation='sigmoid'))
final_model.compile(optimizer=Adam(learning_rate=best_params['learning_rate']),
              loss='binary_crossentropy',
              metrics=['accuracy'])
history = final_model.fit(x_train, y_train1, epochs=100, batch_size=64, validation_data=(x_val, y_val1))
final_model.summary()
_, accuracy_final = final_model.evaluate(x_val, y_val1)

print(f"Best Hyperparameters: {best_params}")
print(f"Validation Accuracy with Best Model: {accuracy_final}")

##############################################################################################

# Plotting Loss and Accuracy after training the final model
plt.figure(figsize=(10, 6))

# Plotting Loss with dots
plt.plot(history.history['loss'], label='Training Loss', color='blue', marker='o', linestyle='-')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange', marker='o', linestyle='-')
plt.xlabel('Epochs')
plt.ylabel('Loss', color='blue')
plt.tick_params(axis='y', labelcolor='blue')
plt.legend(loc='upper left')

# Creating a twin y-axis to plot accuracy
ax2 = plt.gca().twinx()
ax2.plot(history.history['accuracy'], label='Training Accuracy', color='green', marker='o', linestyle='-')
ax2.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red', marker='o', linestyle='-')
ax2.set_ylabel('Accuracy', color='green')
ax2.tick_params(axis='y', labelcolor='green')
ax2.legend(loc='upper right')

plt.title('Training Loss and Accuracy')
plt.xlabel('Epochs')

# Adding grid
plt.grid(True, linestyle='--', alpha=0.7)

# Tight layout to prevent clipping of labels
plt.tight_layout()

plt.show()
