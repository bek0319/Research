import tensorflow as tf
from stock.StockDataBinaryClass import StockDataBinaryClass
from tools.loadFile import loadFile
import numpy as np
import os

# Load data
dir = r"C:\Users\Never nervius\Desktop\18 years"
raw_data1 = loadFile(os.path.join(dir, "EURUSD_diff.csv"))
raw_data1 = raw_data1[28:]
raw_data2 = loadFile(os.path.join(dir, "EURUSD2.csv"))
raw_data2 = raw_data2[29:]
raw_data3 = loadFile(os.path.join(dir, "UPD_EURUSD2.csv"))
features1 = range(5)
features2 = range(5)
features3 = range(10)
prediction = [3]
step = 1
past = 60
future = 1
batch_size = 128

# Create StockDataBinaryClass instances
EURUSD = StockDataBinaryClass(raw_data1, step=step, past=past, future=future, batch_size=batch_size,
                            selected_features=features1, prediction=prediction, split_rate=0.8)

DIFF = StockDataBinaryClass(raw_data2, step=step, past=past, future=future, batch_size=batch_size,
                              selected_features=features2, prediction=prediction, split_rate=0.8)

TI = StockDataBinaryClass(raw_data3, step=step, past=past, future=future, batch_size=batch_size,
                              selected_features=features3, prediction=prediction, split_rate=0.8)

# Get data
x_train, y_train, x_val, y_val = EURUSD.get_npdata()
x_train1, y_train1, x_val1, y_val1 = DIFF.get_npdata()
x_train2, y_train2, x_val2, y_val2 = TI.get_npdata()


# Define model architectures
model_architectures = [
    [
        tf.keras.layers.Conv2D(16, [2,5], data_format="channels_first", padding="same", activation="tanh",
                               kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-4, l2=1e-4)),
        tf.keras.layers.Conv2D(32, [5, 4], data_format="channels_first", padding="same", activation="tanh",
                               kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-4, l2=1e-4)),
        # Add more layers for model 1
    ],
    [
        tf.keras.layers.Conv2D(32, [3,5], data_format="channels_first", padding="same", activation="tanh",
                               kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-5, l2=1e-4)),
        tf.keras.layers.Conv2D(64, [5, 4], data_format="channels_first", padding="same", activation="tanh",
                               kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-5, l2=1e-4)),
        # Add more layers for model 2
    ],
    [
        tf.keras.layers.Conv2D(32, [3,5], data_format="channels_first", padding="same", activation="tanh",
                               kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-5, l2=1e-4)),
        tf.keras.layers.Conv2D(64, [5, 4], data_format="channels_first", padding="same", activation="tanh",
                               kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-5, l2=1e-4)),
        # Add more layers for model 3
    ],
]

# Define datasets and input shapes
datasets = [(x_train, y_train1, x_val, y_val1), (x_train1, y_train1, x_val1, y_val1), ...]
input_shapes = [(x_train.shape[1:], tech_ind_train.shape[1:]),  # Adjust this according to your data
                (x_train1.shape[1:], tech_ind_train.shape[1:]), ...]

# Train models
models = []
for architecture, (x_train, y_train, x_val, y_val), (input_shape, tech_ind_shape) in zip(model_architectures, datasets, input_shapes):
    # Concatenate historical data and technical indicators
    x_train = np.concatenate((x_train, tech_ind_train), axis=-1)
    x_val = np.concatenate((x_val, tech_ind_val), axis=-1)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        *architecture,
        tf.keras.layers.Reshape((past, len(features) + len(tech_ind_train[0]))),
        tf.keras.layers.LSTM(30, dropout=0.2, recurrent_dropout=0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=25, validation_data=(x_val, y_val))
    models.append(model)

# Make predictions and perform voting
predictions = [model.predict(x_val) for model in models]
final_prediction = np.mean(predictions, axis=0) >= 0.5
final_decision = np.where(final_prediction == 1, 'up', 'down')

print("Final Prediction:", final_decision)