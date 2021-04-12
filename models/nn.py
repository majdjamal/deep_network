
"""nn.py: k-Layer Neural Network used for classification, trained with mini batches."""

__author__ = "Majd Jamal"

import numpy as np
from utils.utils import ReLU, softmax

class NN:
    """
    k-Layer Neural Network
    """
    def __init__(self):

        self.W = []   #Weights
        self.b = []   #biases

        self.hidden_units = []
        self.n_layers = None

    def Dense(self, units):
        """Add hidden layer to the network.
        :param units: Number of units in the additional
        hidden layer
        """
        self.hidden_units.append(units)

    def forward(self, X):
        """Computes the forward pass.
        :param X: Data matrix with shape (Ndim, Npts)
        :return
        """
        n_layers = len(self.hidden_units)

        P = None
        H = []

        for i in range(n_layers + 1):

            if i == 0:
                w = self.W[0]
                bias = self.b[0]
                h = ReLU(w@X + bias)
                H.append(h)

            elif i == n_layers:
                h = H[-1]
                w = self.W[-1]
                bias = self.b[-1]
                P = softmax(w @ h + bias)

            else:
                w = self.W[i]
                h = H[i - 1]
                bias = self.b[i]

                h = ReLU(w @ h + bias)
                H.append(h)

        return P, H

    def backward(self, X, Y, P, H, lmd):
        """Computes the backward pass, i.e., gradients.
        :param X: Data matrix with shape (Ndim, Npts)
        :param Y: One hot matrix, shape =(Nout, Npts)
        :param P: Output probabilities, shape = (Nout, Npts)
        :param H: Hidden layer values of various shapes. Inspect forward pass
        to get the exact shapes.
        :param lmd: Regularization constant
        :return gradients_W: Gradients for the weight matricies
        :return gradients_b: Gradients for the bias terms.
        """
        G = - (Y - P)
        _, Npts = P.shape
        n_layers = len(self.hidden_units)

        gradients_W = []
        gradients_b = []

        for i in range(n_layers, -1, -1):

            if i == 0:
                grad_W = G @ X.T * (1/Npts) + 2 * lmd * self.W[i]
                grad_b = G @ np.ones((Npts, 1)) * (1/Npts)

            else:

                h = H[i - 1]
                w = self.W[i]
                grad_W = G @ h.T * (1/Npts) + 2 * lmd * w
                grad_b = G @ np.ones((Npts, 1)) * (1/Npts)

                G = w.T @ G
                G = G * np.where(h > 0, 1, 0)

            gradients_W.append(grad_W)
            gradients_b.append(grad_b)

        return gradients_W, gradients_b


    def update(self, gradients_W, gradients_b, eta):
        """Updates the weights
        :param gradients_W: Gradients for the weight matricies
        :param gradients_b: Gradients for the bias terms.
        :param eta: Learning rate
        """
        for i in range(len(self.W)):

            self.W[i] -= eta * gradients_W[len(self.W) - 1 - i]
            self.b[i] -= eta * gradients_b[len(self.W) - 1 - i]

    def loss(self, X, Y, lmd):
        """ Computes the cross entropy loss.
        :param X: Data matrix with shape (Ndim, Npts)
        :param Y: One hot matrix, shape =(Nout, Npts)
        :param lmd: Regularization constant
        """
        P, _ = self.forward(X)
        loss = np.mean(-np.log(np.einsum('ij,ji->i', Y.T, P)))

        reg = 0   # Regularization term
        for w in self.W:
            reg += np.sum(np.square(w))

        reg *= lmd

        cost = loss + reg

        return cost

    def getParams(self):
        """ Returns weight matrices and bias terms. """
        return self.W, self.b

    def BatchCreator(self, j, n_batch):
        """Creates indices for a mini_batch, given
        an iteration state and number of data points in a batch.
        :param j: Iteration index
        :param n_batch: Number of data points in a batch.
        :return ind: Data point indices to create a mini_batch
        """
        j_start = (j-1)*n_batch + 1
        j_end = j*n_batch + 1
        ind = np.arange(start= j_start, stop=j_end, step=1)
        return ind

    def fit(self, X, Y, y, params):
        """ Train the Neural Network.
        :param X: Data matrix with shape = (Ndim, Npts)
        :param Y: One hot matrix, shape = (Nout, Npts)
        :param y: Labels, shape = (Npts,)
        :param params: Parameters for training, i.e.,
        epochs, n_batch, eta, lmd.
        Note: n_batch is the number of data points in a mini_batch
        """
        Ndim, Npts = X.shape
        Nout, _ = Y.shape
        epochs = params.epochs
        n_batch = params.n_batch
        eta = params.eta
        lmd = params.lmd

        n_layers = len(self.hidden_units)

        # Initialize weights and biases
        if n_layers == 0:
            print('Add Hidden Layers with .Dense(N_units)')

        else:

            for i in range(n_layers + 1):

                if i == 0:
                    n_units = self.hidden_units[i]

                    W_curr = np.random.normal(0, 1/np.sqrt(Ndim),
                    size = (n_units, Ndim))

                    b_curr = np.zeros((n_units, 1))

                elif i == n_layers:
                    n_units_previous = self.hidden_units[i - 1]

                    W_curr = np.random.normal(0, 1/np.sqrt(n_units_previous),
                    size = (Nout, n_units_previous))

                    b_curr = np.zeros((Nout, 1))

                else:
                    n_units = self.hidden_units[i]
                    n_units_previous = self.hidden_units[i - 1]

                    W_curr = np.random.normal(0, 1/np.sqrt(n_units_previous),
                    size = (n_units, n_units_previous))

                    b_curr = np.zeros((n_units, 1))


                self.W.append(W_curr)
                self.b.append(b_curr)

        print('=-=- Training starting -=-= \n')
        for i in range(epochs):

            for j in range(round(Npts/n_batch)):

                #Create mini_batch
                ind = self.BatchCreator(j, n_batch)
                XBatch = X[:, ind]
                YBatch = Y[:, ind]

                #Training
                P, H = self.forward(XBatch)
                gradients_W, gradients_b = self.backward(XBatch, YBatch, P, H, lmd)
                self.update(gradients_W, gradients_b, eta)

            print('=-=- Epoch: ', i, ' -=-=')

        print('\n =-=- Training completed! -=-=')

    def predict(self, X):
        """ Predicts classes for unseen data.
        :return out: predicted classes with shape = (Npts, 1)
        """
        P, _ = self.forward(X)
        out = np.argmax(P, axis=0).reshape(-1,1)
        return out

    def accuracy(self, X, Y, y):
        """ Computes classification accuracy for a given data set.
        :param X: Data matrix with shape = (Ndim, Npts)
        :param Y: One hot matrix, shape = (Nout, Npts)
        :param y: Labels, shape = (Npts,)
        :return: Accuracy rate, constant
        """
        P, _ = self.forward(X)
        out = np.argmax(P, axis=0).reshape(-1,1)
        return np.mean(np.where(y==out, 0, 1))
