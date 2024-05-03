from tools import print_attributes as prt
from tools import findTrendCategory as fTC
import numpy as np
from stock.Normalizor import Normalizor
from tensorflow import keras

class StockData:
    def __init__(self, raw_data, step, past, future, batch_size,
                 selected_features, prediction,
                 split_rate=0.8):
        self.step = step
        self.past = past
        self.future = future
        self.batch_size = batch_size
        self.raw_data =raw_data
        self.selected_features = selected_features
        self.prediction = prediction
        self.split_rate = split_rate


        # The length of a sample equals to the first prediction
        # E.g., past = 12, step = 3, future = 2
        # [0, 3, ..., 30, 33], 34, 35
        self.sample_length = (past-1)*step + 1 + future

        # derive feature_data and prediction from the raw data
        features_data = raw_data[:, selected_features]
        prediction_data = raw_data[:, prediction]
        print("features data shape:", features_data.shape)
        print("prediction data shape:", prediction_data.shape)

        # split data
        split = int(split_rate * len(features_data))
        self.train_x = features_data[:split]
        self.train_y = prediction_data[:split]
        self.val_x = features_data[split:]
        self.val_y = prediction_data[split:]
        # self.val_x = features_data[:split]
        # self.val_y = prediction_data[:split]
        # self.train_x = features_data[split:]
        # self.train_y = prediction_data[split:]
        print("train_data", self.train_x.shape)
        print("val_data", self.val_x.shape)
    # -------------------------------------------
        # create normalizors
        self.train_x_Normalizor = Normalizor(self.train_x)
        self.train_y_Normalizor = Normalizor(self.train_y)
        self.val_x_Normalizor = Normalizor(self.train_x)  # Using the same normalizor
        self.val_y_Normalizor = Normalizor(self.train_y)
        self.val_x_Normalizor = Normalizor(self.val_x)  # Using the different normalizor
        self.val_y_Normalizor = Normalizor(self.val_y)

        # normalize the input data
        self.train_x = self.train_x_Normalizor.normalize(self.train_x)
        self.val_x = self.val_x_Normalizor.normalize(self.val_x)
    #
    #
    #     #
    #     # Don't normalize the output; do that according to the type of the problem
    #     # self.train_y = self.train_y_Normalizor.normalize(self.train_y)
    #     # self.val_y = self.val_y_Normalizor.normalize(self.val_y)
    #
    # # This function returns the data in numpy
    # def get_original_data(self):
    #     # build sequences and targets for training
    #     offset = self.sample_length - 1
    #     x_train = self.train_x #[:-offset]
    #     y_train = self.train_y_Normalizor.normalize(self.train_y)[offset:]
    #
    #     # print("x_train", x_train.shape)
    #     # print("y_train", y_train.shape)
    #
    #     # build sequences and targets for test
    #     x_val = self.val_x #[:-offset]
    #     y_val = self.val_y_Normalizor.normalize(self.val_y)[offset:]
    #     # print("x_val", x_val.shape)
    #     # print("y_val", y_val.shape)
    #
    #     return x_train, y_train, x_val, y_val
    #
    def form_sequences(self, d):
        num_sequence = len(d) - self.sample_length + 1
        x = np.zeros([num_sequence, self.past, d.shape[1]])
        for i in range(num_sequence):
            ids = [j for j in range(i, i + self.past*self.step, self.step)]
            x[i, :] = d[ids]

        return x

    def get_number(self):
        train_y = self.train_y
        train_x = self.train_x
        val_x = self.val_x
        val_y = self.val_y
        x_train = self.form_sequences(train_x)
        x_val = self.form_sequences(val_x)
        y_train = train_y[self.past:]
        y_val = val_y[self.past:]
        return x_train, y_train, x_val, y_val


    #-------------------------------
    #
    # # This function returns the normalized data in numpy
    def get_npdata(self):
        # x_train, y_train = self.form_sequences(self.train_x, self.train_y)
        # x_val, y_val = self.form_sequences(self.val_x, self.val_y)
        x_train, y_train, x_val, y_val = self.get_original_data()
        x_train = self.form_sequences(x_train)
        x_val = self.form_sequences(x_val)
        print("shape: x_train -", x_train.shape, ", y_train -", y_train.shape)
        print("shape: x_val -", x_val.shape, ", y_val -", y_val.shape)

        return x_train, y_train, x_val, y_val
    #
    # # This function returns the normalized data in tf detaset
    def get_tfdata(self):
        x_train, y_train, x_val, y_val = self.get_original_data()

        # generate tensorflow datasets
        dataset_train = keras.preprocessing.timeseries_dataset_from_array(
            x_train, y_train,
            sequence_length=self.past,
            sequence_stride=self.future,
            sampling_rate=self.step,
            batch_size=self.batch_size
        )

        dataset_val = keras.preprocessing.timeseries_dataset_from_array(
            x_val, y_val,
            sequence_length=self.past,
            sequence_stride=self.future,
            sampling_rate=self.step,
            batch_size=self.batch_size
        )

        return dataset_train, dataset_val
    #
    # # This function resumes to the original target values
    # def denormalize(self, targets, category):
    #     if category == 'train':
    #         return self.train_y_Normalizor.denormalize(targets)
    #     else:
    #         return self.val_y_Normalizor.denormalize(targets)
    #
    # # This function finds the common-sense prediction and mse loss
    # # That is, target is predicted as the same as the last value of the sequence
    # def get_baseline_prediction(self):
    #     offset = self.sample_length - 1
    #     y_train_prediction = self.train_y_Normalizor.normalize(self.train_y)[offset-1: -1]  # a naive prediciton
    #     y_val_prediction = self.val_y_Normalizor.normalize(self.val_y)[offset-1: -1]  # a naive prediciton
    #
    #     # return the common-sense prediction
    #     return y_train_prediction, y_val_prediction
    #
    # def compare_stock_prices(self, data):
    #     compared_data = np.zeros((len(data) - 1, data.shape[1]))
    #
    #     for i in range(len(data) - 1):
    #         for j in range(data.shape[1]):
    #             compared_data[i, j] = 1 if data[i + 1, j] >= data[i, j] else 0
    #
    #     return compared_data
    #
    # def get_binary(self):
    #     offset = self.sample_length -1
    #     compared_data = self.compare_stock_prices(self.raw_data)
    #     features_data = compared_data[:, self.selected_features]
    #     prediction_data = compared_data[:, self.prediction]
    #     split = int(self.split_rate * len(features_data))
    #     train_y = self.train_y
    #     train_x = features_data[:split]
    #     val_x = features_data[split:]
    #     val_y = self.val_y
    #     x_train = self.form_sequences(train_x)
    #     x_val = self.form_sequences(val_x)
    #
    #     train_y1 = train_y[offset:]
    #     val_y1 = val_y[offset -1:]
    #     y_train = fTC.get_trend(train_y1)
    #     y_val = fTC.get_trend(val_y1)
    #     # y_train = train_y[offset - 1:]
    #     # y_val = val_y[offset-1:]
    #     return x_train, y_train, x_val, y_val
    #
