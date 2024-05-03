import numpy as np

# find the trends in a single array
# 1 if data[i+1] >= data[i], i=0,1,2,...,n-2
# 0 otherwise
# so the length of the resulting array is n-1
# OR
# compare two arrays element wisely and find the trends
# def get_trend(data, ref=None):
#     if ref is None:
#         return np.array([[1.0] if data[i, 0] >= data[i - 1, 0] else [0.0] for i in range(1, len(data))])
#     else:
#         return np.array([[1.0] if data[i, 0] >= ref[i, 0] else [0.0] for i in range(0, len(data))])


# TO compare 7 days later
def get_trend(data, ref=None):
    if ref is None:
        return np.array([[1.0] if data[i, 0] <= data[i + 3, 0] else [0.0] for i in range(len(data) - 3)])
    else:
        return np.array([[1.0] if data[i, 0] <= ref[i - 3, 0] else [0.0] for i in range(3, len(data))])

# compute the percentage of correct predictions

def get_accuracy(prediction_trend, real_trend):
    count = 0
    n = len(prediction_trend)
    for i in range(n):
        if np.equal(prediction_trend[i], real_trend[i]):
            count += 1
    return float(count)/float(n)

# get distributions/histograms
def get_binaryclass_distribution(data):
    num_classes = 2  # a binary-class problem
    count = np.asarray([0., 0.])
    v = np.asarray([[0.], [1.]])

    for i in range(len(data)):
        for j in range(num_classes):
            if np.equal(data[i], v[j]):
                count[j] += 1
    return count/float(len(data))

# get distributions/histograms
def get_multiclass_distribution(data):
    num_classes = len(data[0])
    count = np.asarray([0.]*num_classes)
    v = np.asarray([[0.]*num_classes]*num_classes)
    for i in range(num_classes):
        v[i, len(v)-i-1] = 1.0

    for i in range(len(data)):
        for j in range(num_classes):
            if (data[i] == v[j]).all():
                count[j] += 1.
    return count/float(len(data))

# The targets predict the trend of the close price, for example,
# [1,0,0,0] for >=3%
# [0,1,0,0] for >=0%
# [0,0,1,0] for >=-3%
# [0,0,0,1] for <-3%
def get_category_vector(curr, prev, th):
    n = len(th) + 1
    v = [0.0]*n

    percent = (curr - prev) / prev
    for i in range(n-1):
        if percent >= th[n-i-2]:
            v[i] = 1.0  
            return v
        
    v[n-1] = 1.0  # the default vector
    return v

def get_category(th, data, ref=None):
    if ref is None:
        return np.array([get_category_vector(data[i, 0], data[i-1, 0], th) for i in range(1, len(data))])
    else:
        return np.array([get_category_vector(data[i, 0], ref[i, 0], th) for i in range(0, len(data))])

# def get_sparse_category(prev, curr):
#     percent = (curr - prev) / prev
#     if percent >= 0.03:
#         return 0.0
#     elif percent >= 0.0:
#         return 1.0
#     elif percent >= -0.03:
#         return 2.0
#     else:
#         return 3.0