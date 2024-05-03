import unittest
from tools.augmentData import *

class TestShuffleColumns(unittest.TestCase):
    def test_shuffle_columns(self):
        arr1 = np.asarray([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
        shuffle_columns(arr1, [2,1,0])
        arr2 = np.asarray([[3,2,1],[6,5,4],[9,8,7],[12,11,10]])
        self.assertTrue((arr1==arr2).all())

        arr1 = np.asarray([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20]])
        shuffle_columns(arr1, [2,1,3,4,0])
        arr2 = np.asarray([[3,2,4,5,1],[8,7,9,10,6],[13,12,14,15,11],[18,17,19,20,16]])
        self.assertTrue((arr1==arr2).all())

        




 