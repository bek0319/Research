import unittest
from stock.StockData import StockData
from tools.loadFile import loadFile

class TestStockData (unittest.TestCase):
    def compare(self, tfset, xset, yset):
        sample_count = 0
        batch_count = 0        
        batch_size = 0
        for batch in tfset:
            # get a batch
            # print("batch", count, ":")
            x, y = batch

            # convert it to numpy
            x = x.numpy()
            y = y.numpy()

            # compute the range
            if batch_count == 0:
                batch_size = len(x)
            start = batch_count * batch_size
            end = start + len(x)

            # compare with x_train in a elementwise manner
            comparison = x == xset[start: end]
            self.assertTrue(comparison.all())

            # compare with y_train in a elementwise manner
            comparison = y == yset[start: end]
            self.assertTrue(comparison.all())

            sample_count += len(x)

            # increment
            batch_count += 1

        print("Total samples:", sample_count)
        print("Total batchs:", batch_count)

    def test_compare_npdata_and_tfdata(self):
        dir = "D:\\Github\\workplace\\20230514_5y\\"
        raw_data = loadFile(dir + "GOOGL.csv")

        # Available features include ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        features = [0, 1, 2, 3, 5]
        prediction = [3]

        step = 1  # sample rate
        past = 30  # look backward into the past, will be the sequence length
        future = 1  # predict the value in the future; 1 means right after, will be the sequence stride
        batch_size = 16
        GOOG = StockData(raw_data, step=step, past=past, future=future, batch_size=batch_size,
                        selected_features=features, prediction=prediction,
                        split_rate=0.8)

        # Compare two methods for preparing data
        dataset_train, dataset_val = GOOG.get_tfdata()
        x_train, y_train, x_val, y_val = GOOG.get_npdata()

        self.compare(dataset_train, x_train, y_train)
        self.compare(dataset_val, x_val, y_val)
 