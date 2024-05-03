import os

import numpy as np


def logitstic_loss(true_labels, predictions, sparse=False, logit=False):
    true_labels = np.array(true_labels)
    if sparse:
        num_classes = len(predictions[0])
        num_labels = len(true_labels)
        x = np.zeros(shape=(num_labels, num_classes))
        for i in range(num_labels):
            x[i, true_labels[i]] = 1
        true_labels = x
        # print(true_labels)

    predictions = np.array(predictions)
    if logit:
        predictions = np.exp(predictions)
    sum = predictions.sum(axis=1)
    for i in range(len(predictions)):
        predictions[i, :] /= sum[i]
    # print(predictions)

    loss = - (true_labels * np.log(predictions))
    return loss.sum(axis=1)
