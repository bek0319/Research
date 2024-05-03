import numpy as np

# shuffle the columns of a 2d array
def shuffle_columns(data, idx=None):
    num_col = len(data[0])
    if idx is None:
        idx = np.asarray(range(num_col))
        np.random.shuffle(idx)

    for i in range(len(data)):
        data[i] = data[i][idx]
    