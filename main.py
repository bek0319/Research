import tensorflow as tf
print("tensorflow", tf. __version__)
gpus = tf.config.list_physical_devices()
for gpu in gpus:
    print("Name:", gpu.name, "  Type:", gpu.device_type)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print("Done!")

# import stock.lstm_stock
# import stock.lstm_stock_binaryclass
import stock.lstm_stock_multiclass