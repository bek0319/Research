import stock.StockData as sd
from tensorflow import keras
import numpy as np

class StockDataMultiClass(sd.StockData):
    # This function returns the normalized training and test dataset for tensorflow
    # The targets predict the trend of close prices, for example,
    # [1,0,0,0] for >=3%
    # [0,1,0,0] for >=0%
    # [0,0,1,0] for >=-3%
    # [0,0,0,1] for <-3%
    def get_dataset(self):
        # split the data
        train_data = self.features_data[:self.split]
        val_data = self.features_data[self.split:]

        # generate tensorflow datasets
        offset = self.past + self.future - 1
        x_train = train_data[:-self.future]
        y_train = train_data[offset-1:, self.target_feature_index]
        y_train = np.array([self.get_sparse_category(y_train[i-1, 0], y_train[i, 0]) for i in range(1, y_train.shape[0])])
        x_train = self.normalizor_train.normalize(x_train)
        dataset_train = keras.preprocessing.timeseries_dataset_from_array(
            x_train, y_train,
            sequence_length=self.past,
            batch_size=self.batch_size,
        )

        x_val = val_data[:-self.future]
        y_val = val_data[offset-1:, self.target_feature_index]
        y_val = np.array([self.get_sparse_category(y_val[i-1, 0], y_val[i, 0]) for i in range(1, y_val.shape[0])])
        x_val = self.normalizor_test.normalize(x_val)
        dataset_val = keras.preprocessing.timeseries_dataset_from_array(
            x_val, y_val,
            sequence_length=self.past,
            batch_size=self.batch_size,
        )

        return dataset_train, dataset_val

    def get_category(self, prev, curr):
        percent = (curr - prev) / prev
        if percent >= 0.05:
            return [1.0, 0.0, 0.0, 0.0]
        elif percent >= 0.0:
            return [0.0, 1.0, 0.0, 0.0]
        elif percent >= -0.05:
            return [0.0, 0.0, 1.0, 0.0]
        else:
            return [0.0, 0.0, 0.0, 1.0]

    def get_sparse_category(self, prev, curr):
        percent = (curr - prev) / prev
        if percent >= 0.05:
            return 0
        elif percent >= 0.0:
            return 1
        elif percent >= -0.05:
            return 2
        else:
            return 3

    # This function returns the normalized training dataset in numpy array format
    # The targets predict the trend of close prices, either 1 for increasing or 0 for decreasing
    def get_data_train(self):
        data_train = self.features_data[:self.split]
        data_train = self.normalizor_train.normalize(data_train)
        offset = self.past + self.future - 1
        num_targets = data_train.shape[0] - offset

        x_train = np.array([])
        y_train = np.array([])
        for i in range(num_targets):
            x_train = np.append(x_train, data_train[i:i+self.past, :])
            prev = data_train[i+offset-1, self.target_feature_index[0]]
            curr = data_train[i+offset, self.target_feature_index[0]]
            y_train = np.append(y_train, self.get_sparse_category(prev, curr))

        x_train = x_train.reshape(-1, self.past, data_train.shape[1])
        y_train = y_train.reshape(-1, 1)
        return x_train, y_train

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
                    print("Error at X", i, j)
                    print(X[i * self.past + j, :].tolist())
                    print(data_original[i + j, :].tolist())
                    return False
            prev = data_original[i+offset-1, self.target_feature_index[0]]
            curr = data_original[i+offset, self.target_feature_index[0]]
            prev = self.normalizor_train.denormalize(prev)[0]
            curr = self.normalizor_train.denormalize(curr)[0]
            if (y[i] != self.get_sparse_category(prev, curr)):
                print("Error at y", i)
                print(y[i])
                print(prev, curr)
                return False

        return True