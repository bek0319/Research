import math
import optuna
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import layers
from sklearn.metrics import mean_squared_error
from keras.layers import Conv2D, MaxPooling2D, Flatten, LSTM, RepeatVector, BatchNormalization


data1 = pd.read_csv(r"C:\Users\Never nervius\Desktop\18 years\EURUSD.csv")
data1 = data1.drop(['Date'], axis=1)
data2 = pd.read_csv(r"C:\Users\Never nervius\Desktop\18 years\EURGBP.csv")
data2 = data2.drop(['Date'], axis=1)
data = np.concatenate((data1, data2), axis=1)

values = data
training_data_length = math.ceil(len(values)*0.8)
values_training = values[:training_data_length]
values_testing = values[training_data_length:]
scaler = MinMaxScaler(feature_range = (0,1))
scaled_data_training = scaler.fit_transform(values_training.reshape(-1, 10))
scaled_data_testing = scaler.fit_transform(values_testing.reshape(-1, 10))

X_train = []
y_train = []

for i in range(60, training_data_length):
  X_train.append(scaled_data_training[i-60:i, :])
  y_train.append(values_training[i, 3])

X_train, y_train = np.array(X_train), np.array(y_train)

#Testing data
X_test = []
y_test = []

for i in range(60, len(values_testing)):
  X_test.append(scaled_data_testing[i -60: i, :])
  y_test.append(values_testing[i, 3])
X_test, y_test = np.array(X_test), np.array(y_test)

def objective(trial):
    # Define the search space
    cnn_filter1 = trial.suggest_int('cnn_filter1', 10, 250)
    cnn_filter2 = trial.suggest_int('cnn_filter2', 10, 256)
    cnn_filter3 = trial.suggest_int('cnn_filter3', 10, 256)
    cnn_filter4 = trial.suggest_int('cnn_filter4', 10, 256)
    cnn_filter5 = trial.suggest_int('cnn_filter5', 10, 256)
    # lstm1 = trial.suggest_int('lstm1', 50, 250)
    lstm2 = trial.suggest_int('lstm2', 50, 250)
    lstm3 = trial.suggest_int('lstm3', 20, 250)

    model = keras.Sequential()
    model.add(Conv2D(cnn_filter1, (3, 3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1), padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(cnn_filter2, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(cnn_filter3, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(cnn_filter4, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 1)))
    model.add(BatchNormalization())
    model.add(Conv2D(cnn_filter5, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 1)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(RepeatVector(1))
    # model.add(LSTM(lstm1, activation='relu', return_sequences=True))
    model.add(LSTM(lstm2, activation='relu', return_sequences=True))
    model.add(LSTM(lstm3, activation='relu', return_sequences=False))
    model.add(layers.Dense(64))
    model.add(layers.Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32, verbose=2)

    val_loss = history.history['val_loss'][-1]

    return val_loss

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

best_params = study.best_params

print("Best Hyperparameters:", best_params)

# Re-train the model with the best hyperparameters
best_model = keras.Sequential()
best_model.add(Conv2D(best_params['cnn_filter1'], (3, 3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1), padding='same'))
best_model.add(MaxPooling2D((2, 2)))
best_model.add(BatchNormalization())
best_model.add(Conv2D(best_params['cnn_filter2'], (3, 3), activation='relu', padding='same'))
best_model.add(MaxPooling2D((2, 2)))
best_model.add(BatchNormalization())
best_model.add(Conv2D(best_params['cnn_filter3'], (3, 3), activation='relu', padding='same'))
best_model.add(MaxPooling2D((2, 2)))
best_model.add(BatchNormalization())
best_model.add(Conv2D(best_params['cnn_filter4'], (3, 3), activation='relu', padding='same'))
best_model.add(MaxPooling2D((2, 1)))
best_model.add(BatchNormalization())
best_model.add(Conv2D(best_params['cnn_filter5'], (3, 3), activation='relu', padding='same'))
best_model.add(MaxPooling2D((2, 1)))
best_model.add(BatchNormalization())
best_model.add(Flatten())
best_model.add(RepeatVector(1))
# best_model.add(LSTM(best_params['lstm1'], activation='relu', return_sequences=True))
best_model.add(LSTM(best_params['lstm2'], activation='relu', return_sequences=True))
best_model.add(LSTM(best_params['lstm3'], activation='relu', return_sequences=False))
best_model.add(layers.Dense(64))
best_model.add(layers.Dense(1))

best_model.compile(optimizer='adam', loss='mean_squared_error')

history = best_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=200, batch_size=32, verbose=2)


predictions_train = best_model.predict(X_train)
predictions_train = np.squeeze(predictions_train, axis=-1)
predictions_val = best_model.predict(X_test)
predictions_val = np.squeeze(predictions_val, axis=-1)

# Plotting loss
plt.figure(figsize=(16, 12), dpi=100)
plt.subplot(3, 1, 1)
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
plt.legend()

#Plotting training data
plt.subplot(3, 1, 2)
plt.title('Training Data')
plt.xlabel('Days')
plt.ylabel('Price')
plt.plot(y_train, label='True Values', color='blue')
plt.plot(predictions_train.reshape(-1), label='Training Predictions', color='red')
plt.legend()

# Plotting validation data
plt.subplot(3, 1, 3)
plt.title('Validation Data')
plt.xlabel('Days')
plt.ylabel('Price')
plt.plot(y_test, label='True Values', color='green')
plt.plot(predictions_val.reshape(-1), label='Predictions', color='red')
plt.legend()

plt.tight_layout()
plt.show()

# Calculating MSE
mse_train = mean_squared_error(y_train, predictions_train)
mse_val = mean_squared_error(y_test, predictions_val)
print("Training MSE:", "{:.10f}".format(mse_train))
print("Validation MSE:", "{:.10f}".format(mse_val))
