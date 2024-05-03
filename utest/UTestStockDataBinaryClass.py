import stock.StockData as sd
from tensorflow import keras
import numpy as np

class StockDataBinaryClass(sd.StockData):
    # This function returns the normalized training and test dataset for Tensorflow
    # The targets predict the trend of close prices, 1 for increasing and 0 for decreasing
    def get_npdata(self):
        x_train = self.train_data
        x_train = self.train_Normalizor.normalize(x_train)

        offset = self.past + self.future - 1
        y_train = self.train_data[offset-1:, self.target_index]
        print('y_train shape', y_train.shape)

        # note an extra values is included by offset-1 above
        # build up the trend as the output
        y_train = np.array([[1.0] if y_train[i, 0] >= y_train[i-1, 0] else [0.0] for i in range(1, y_train.shape[0])])
        print('y_train shape', y_train.shape)

        dataset_train = keras.preprocessing.timeseries_dataset_from_array(
            x_train, y_train,
            sequence_length=self.past,
            batch_size=self.batch_size
        )

        x_val = self.val_data
        x_val = self.val_Normalizor.normalize(x_val)
        y_val = self.val_data[offset-1:, self.target_index]
        y_val = np.array([[1.0] if y_val[i, 0] >= y_val[i-1, 0] else [0.0] for i in range(1, y_val.shape[0])])

        dataset_val = keras.preprocessing.timeseries_dataset_from_array(
            x_val, y_val,
            sequence_length=self.past,
            batch_size=self.batch_size
        )

        return dataset_train, dataset_val

    # This function returns the original (non-normalized) training data
    def get_original_data_train(self):
        # build sequences and targets for training
        offset = self.past + self.future - 1
        x_train = self.train_data
        y_train = self.train_data[offset-1:, self.target_index]
        y_train = np.array([[1.0] if y_train[i, 0] >= y_train[i - 1, 0] else [0.0] for i in range(1, y_train.shape[0])])

        return x_train, y_train

    # This function returns the original (non-normalized) test data
    def get_original_data_test(self):
        # build sequences and targets for test
        offset = self.past + self.future - 1
        x_val = self.val_data
        y_val = self.val_data[offset-1:, self.target_index]
        y_val = np.array([[1.0] if y_val[i, 0] >= y_val[i - 1, 0] else [0.0] for i in range(1, y_val.shape[0])])

        return x_val, y_val

    # # This function returns the normalized training dataset in numpy array format
    # # The targets predict the trend of close prices, either 1 for increasing or 0 for decreasing
    # def get_data_train(self):
    #     data_train = self.features_data[:self.split]
    #     data_train = self.normalizor_train.normalize(data_train)
    #     offset = self.past + self.future - 1
    #     num_targets = data_train.shape[0] - offset
    #
    #     x_train = np.array([])
    #     y_train = np.array([])
    #     for i in range(num_targets):
    #         x_train = np.append(x_train, data_train[i:i+self.past, :])
    #         prev = data_train[i+offset-1, self.target_feature_index[0]]
    #         curr = data_train[i+offset, self.target_feature_index[0]]
    #         y_train = np.append(y_train, 1.0 if curr >= prev else 0.0)
    #
    #     x_train = x_train.reshape(-1, self.past, data_train.shape[1])
    #     y_train = y_train.reshape(-1, 1)
    #     return x_train, y_train

    # dataset is a Tensorflow tensor
    def UTEST_dataset_train(self, dataset):
        data_original = self.features_data[:self.split]
        data_original = self.normalizor_train.normalize(data_original)

        X = np.array([])
        y = np.array([])
        for batch in dataset.__iter__():
            inputs, targets = batch
            X = np.append(X, inputs.numpy())
            y = np.append(y, targets.numpy())

        X = X.reshape(-1, data_original.shape[1])
        y = y.reshape(-1, 1)

        print(data_original.shape)
        print(X.shape, y.shape)

        offset = self.past + self.future - 1
        num_targets = data_original.shape[0] - offset
        if (X.shape[0] != self.past * num_targets) or (y.shape[0] != num_targets):
            return False

        for i in range(num_targets):
            for j in range(self.past):
                if X[i * self.past + j, :].tolist() != data_original[i + j, :].tolist():
                    print(i, j)
                    print(X[i * self.past + j, :].tolist())
                    print(data_original[i + j, :].tolist())
                    return False
            prev = data_original[i+offset-1, self.target_feature_index[0]]
            curr = data_original[i+offset, self.target_feature_index[0]]
            if (y[i] == 1.0 and curr < prev) or (y[i] == 0.0 and curr >= prev):
                print(i)
                print(y[i])
                print(prev, curr)
                return False

        return True