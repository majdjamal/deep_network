
import numpy as np
import matplotlib.pyplot as plt


class Params:

    def __init__(self, epochs, n_batch, eta, lmd):

        self.epochs = epochs
        self.n_batch = n_batch
        self.eta = eta
        self.lmd = lmd

def centering(X):
    """Transform data matrix to have zero mean.
    :return X: Data matrix, shape = (Ndim, Npts)
    :return X: Data matrix with zero mean.
    """
    Ndim, Npts = X.shape

    mu = np.mean(X, axis=1).reshape(-1,1)
    sig = np.std(X, axis=1).reshape(-1,1)

    X = X - mu
    X = X / sig

    return X

def softmax(x):
    """ Standard definition of the softmax function """
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def ReLU(x):
    """ Standard definition of the ReLU function """
    return np.maximum(x, 0)

