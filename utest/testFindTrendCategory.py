import unittest
from tools.findTrendCategory import *

class TestFindTrendCategory (unittest.TestCase):
    def test_get_trend_accuracy(self):
        real    = [1,2,3,4,5,6,7,8,9,10]
        real    = np.reshape(real, [-1,1])
        predict = [3,5,7,8,4,2,3,1,0,11]
        predict = np.reshape(predict, [-1,1])

        t1 = get_trend(real)
        t2 = np.reshape([1., 1., 1., 1., 1., 1., 1., 1., 1.], [-1,1])
        self.assertTrue((t1==t2).all())

        t3 = get_trend(predict[1:], real[:-1])
        t4 = np.reshape([1., 1., 1., 1., 0., 0., 0., 0., 1.], [-1,1])
        self.assertTrue((t3==t4).all())

        self.assertTrue(get_accuracy(t3, t1) == 5.0/9.0)

    def test_get_category(self):
        prev = [1.0, 2.0, 3.0, 4.0, 5.0] #, 6.0, 7.0, 8.0, 9.0, 10.0]
        prev = np.reshape(prev, [-1,1])
        curr = [3.0, 1.8, 3.01, 4.1, 4.99] #, 2.9, 7.1, 1.0, 0.9, 11.0]
        curr = np.reshape(curr, [-1,1])

        th = [-0.03, 0.0, 0.03]
        cate = [[1,0,0,0], [0,0,0,1], [0,1,0,0], [0,1,0,0], [0,0,1,0]]
        cate = np.asarray(cate)
        cate = cate.astype(float)

        cate1 = get_category(th, curr, prev)
        self.assertTrue((cate1 == cate).all())

        count = get_multiclass_distribution(cate)
        self.assertTrue((count == [1./5., 1./5., 2./5., 1./5.]).all())



 