import stock.StockData as sd
from tensorflow import keras
import numpy as np
from tools.findTrendCategory import *
from tools.augmentData import shuffle_columns

class StockDataMultiClass(sd.StockData):
    class_division = [-0.03, 0.0, 0.03]  # default multi classes

    # Override
    # The targets predict the trend of the close price, for example,
    # [1,0,0,0] for >=3%
    # [0,1,0,0] for >=0%
    # [0,0,1,0] for >=-3%
    # [0,0,0,1] for <-3%
    def get_original_data(self):
        # build sequences and targets for training
        offset = self.sample_length - 1
        x_train = self.train_x
        y_train = self.train_y[offset-1:]  # offset-1: need one more price to find n trends
        y_train = get_category(self.class_division, y_train)
        # print("x_train", x_train.shape)
        # print("y_train", y_train.shape)

        # build sequences and targets for test
        x_val = self.val_x
        y_val = self.val_y[offset-1:]
        y_val = get_category(self.class_division, y_val)
        # print("x_val", x_val.shape)
        # print("y_val", y_val.shape)
        
        return x_train, y_train, x_val, y_val
    
    def get_number_classes(self):
        return len(self.class_division)+1
    
    def set_class_division(self, th):
        self.class_division = th

    # Override
    def get_npdata_augmented(self):
        # x_train, y_train = self.form_sequences(self.train_x, self.train_y)
        # x_val, y_val = self.form_sequences(self.val_x, self.val_y)
        x_train, y_train, x_val, y_val = self.get_original_data()

        # make copies for data augmentation
        x_train_1 = x_train
        x_val_1 = x_val

        # reorder the features per columns
        idx = np.asarray(range(6*6))
        for i in range(6):
            for j in range(6):
                idx[i*6+j] = j*6+i
        shuffle_columns(x_train_1, idx)
        shuffle_columns(x_val_1, idx)

        # idx = np.asarray(range(6*6))
        # np.random.shuffle(idx)
        # count = 0
        # for id in idx:
        #     if id == 3: break
        #     count += 1
        # shuffle_columns(x_train_1, idx)

        x_train = np.concatenate((x_train, x_train_1), axis=1)
        x_train = self.form_sequences(x_train)

        x_val = np.concatenate((x_val, x_val_1), axis=1)
        x_val = self.form_sequences(x_val)

        print("shape: x_train -", x_train.shape, ", y_train -", y_train.shape)
        print("shape: x_val -", x_val.shape, ", y_val -", y_val.shape)
        
        return x_train, y_train, x_val, y_val
    
    def get_npdata_reordered(self):
        # x_train, y_train = self.form_sequences(self.train_x, self.train_y)
        # x_val, y_val = self.form_sequences(self.val_x, self.val_y)
        x_train, y_train, x_val, y_val = self.get_original_data()

        # reorder the features per columns
        idx = np.asarray(range(6*6))
        for i in range(6):
            for j in range(6):
                idx[i*6+j] = j*6+i
        shuffle_columns(x_train, idx)
        shuffle_columns(x_val, idx)

        # idx = np.asarray(range(6*6))
        # np.random.shuffle(idx)
        # count = 0
        # for id in idx:
        #     if id == 3: break
        #     count += 1
        # shuffle_columns(x_train_1, idx)
 
        x_train = self.form_sequences(x_train)
        x_val = self.form_sequences(x_val)

        print("shape: x_train -", x_train.shape, ", y_train -", y_train.shape)
        print("shape: x_val -", x_val.shape, ", y_val -", y_val.shape)
        
        return x_train, y_train, x_val, y_val