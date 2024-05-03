import stock.StockData as sd
from tensorflow import keras
import numpy as np
from tools.findTrendCategory import *

class StockDataBinaryClass(sd.StockData):
    # Override
    # The targets predict the trend of the close price, 1 for increasing and 0 for decreasing
    def get_original_data(self):
        # x_train = self.train_x
        # y_train = self.train_y[self.past:]
        # x_val = self.val_x
        # y_val = self.val_y[self.past:]
        # build sequences and targets for training
        offset = self.sample_length - 1
        x_train = self.train_x
        y_train = self.train_y[offset-1:]  # offset-1: need one more price to find n trends
        y_train = get_trend(y_train)
        # print("x_train", x_train.shape)
        # print("y_train", y_train.shape)

        # build sequences and targets for test
        x_val = self.val_x
        y_val = self.val_y[offset-1:]
        y_val = get_trend(y_val)
        # print("x_val", x_val.shape)
        # print("y_val", y_val.shape)
        
        return x_train, y_train, x_val, y_val
