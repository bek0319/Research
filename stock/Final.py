import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Conv2D, Reshape, Input, Dense
from keras.optimizers import Adam
from keras import regularizers
from keras import layers

# Defining model parameters
past = 60  # look backward into the past
features = 5
learning_rate = 0.0001
days_ahead = 2  # Predicting 3 days later

# Define the normalization functions
def compute_normalization_params(data):
    a = data.mean(axis=0)
    b = data.std(axis=0)
    return a, b

def normalize_data(data, a, b, indices=None):
    if indices is None:
        return (data - a) / b
    else:
        a = [a[i] for i in indices]
        b = [b[i] for i in indices]
        return (data - a) / b

# Define the function to generate binary labels
def get_trend(data, index_number):
    labels = []
    for i in range(len(data) - days_ahead):
        current_price = data[i, index_number]
        future_price = data[i + days_ahead, index_number]
        if current_price <= future_price:
            labels.append(1)  # Append 1.0 for upward trend
        else:
            labels.append(0)  # Append 0.0 for downward trend
    return np.array(labels)

# Define your parameters and file paths here
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
dir = r"C:\Users\Never nervius\Desktop\18 years"

# Load data
data1 = pd.read_csv(os.path.join(dir, "EURUSD2.csv"))
# data1 = data1[days_pred:]
data1 = data1.drop('Date', axis=1)
data2 = pd.read_csv(os.path.join(dir, "EURUSD2.csv"))
data2 = data2.drop('Date', axis=1)


# Extract values from Difference data
true_values = data1.values
diff_values = data2.values
true_values_length = math.ceil(len(true_values) * 0.8)
diff_values_length = math.ceil(len(diff_values) * 0.8)

true_values_training = true_values[:true_values_length]
true_values_testing = true_values[true_values_length:]
diff_values_training = diff_values[:diff_values_length]
diff_values_testing = diff_values[diff_values_length:]

# Compute normalization parameters for training
a1, b1 = compute_normalization_params(diff_values_training)

# Compute normalization parameters for testing
a2, b2 = compute_normalization_params(diff_values_testing)

# Normalize the training data
normalized_data_training = normalize_data(diff_values_training, a1, b1)
normalized_data_testing = normalize_data(diff_values_testing, a2, b2)

# normalized_data_training = diff_values_training
# normalized_data_testing = diff_values_testing

# Generate binary labels for training and testing data
y_train = get_trend(true_values_training[past+days_ahead-1:], 3)
y_test = get_trend(true_values_testing[past+days_ahead-1:], 3)

# Define X_train, X_test
X_train = []
X_test = []

for i in range(past, len(diff_values_training)):
    X_train.append(normalized_data_training[i - past:i, :])

for i in range(past, len(diff_values_testing)):
    X_test.append(normalized_data_testing[i - past: i, :])

X_train, X_test = np.array(X_train), np.array(X_test)
X_train = X_train[days_ahead+days_ahead-1:]
X_test = X_test[days_ahead+days_ahead -1:]

# Reshaping the data
X_train = X_train.reshape(X_train.shape[0], 1, past, features)
X_test = X_test.reshape(X_test.shape[0], 1, past, features)

# Define and compile the model
model = Sequential()
model.add(Input(shape=(1, past, features)))
model.add(Conv2D(16, [2,5], data_format="channels_first", padding="same", activation="tanh",
                                   kernel_regularizer=regularizers.L1L2(l1=1e-4, l2=1e-4)))
model.add(Conv2D(32, [5, 4], data_format="channels_first", padding="same", activation="tanh",
             kernel_regularizer=regularizers.L1L2(l1=1e-4, l2=1e-4))),
model.add(Conv2D(64, [3, 2], data_format="channels_first", padding="same", activation="tanh",
             kernel_regularizer=regularizers.L1L2(l1=1e-4, l2=1e-4))),
model.add(Conv2D(128, [4, 3], data_format="channels_first", padding="same", activation="tanh",
             kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4))),
model.add(Conv2D(32, [4, 2], data_format="channels_first", padding="same", activation="tanh",
             kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4))),
model.add(Conv2D(1, [2, 4], data_format="channels_first", padding="same", activation="tanh",
             kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4))),
model.add(Reshape((past, features)))
model.add(LSTM(30, dropout=0.2, recurrent_dropout=0.2))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=Adam(learning_rate=learning_rate),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()


# Train the model
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))

# Plotting Accuracy and Loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss', color='blue', marker='o', linestyle='-')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange', marker='o', linestyle='-')
plt.xlabel('Epochs')
plt.ylabel('Loss', color='blue')
plt.tick_params(axis='y', labelcolor='blue')
plt.legend(loc='upper left')
ax2 = plt.gca().twinx()
ax2.plot(history.history['accuracy'], label='Training Accuracy', color='green', marker='o', linestyle='-')
ax2.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red', marker='o', linestyle='-')
ax2.set_ylabel('Accuracy', color='green')
ax2.tick_params(axis='y', labelcolor='green')
ax2.legend(loc='upper right')
plt.title('Training Loss and Accuracy')
plt.xlabel('Epochs')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()