
import numpy as np
from data.data import LoadBatch
from utils.utils import centering, softmax, ReLU, Params
from models.nn import NN

#=-=-=-=-=-=-=-
# Experiments
#=-=-=-=-=-=-=-

#=-=-=-=-=-=-=-=-
# 1. Load Data
#=-=-=-=-=-=-=-=-

print('=-=- Loading data -=-= \n')
X_train, Y_train, y_train = LoadBatch('data/data_batch_1')
X_val, Y_val, y_val = LoadBatch('data/data_batch_2')
X_test, Y_test, y_test = LoadBatch('data/test_batch')

# OBS! You can pass your own data, as long as it has these dimensions:
# X.shape = (Ndim, Npts)
# Y.shape = (Nout, Npts), which is one hot vector matrix. See data.py to know
# how to generate it from labels.
# y.shape = (Npts, )

#=-=-=-=-=-=-=-=-=-=-=-=-
# 2. Pre-processing.
#=-=-=-=-=-=-=-=-=-=-=-=-

X_train = centering(X_train)
X_val = centering(X_val)
X_test = centering(X_test)
print('=-=- Data loading is completed! -=-= \n')

#=-=-=-=-=-=-=-=-=-=-=-=-
# 3. Params.
#=-=-=-=-=-=-=-=-=-=-=-=-

params = Params(
	epochs = 10, n_batch = 100, eta = 0.01, lmd = 0.0
	)

#=-=-=-=-=-=-=-=-=-=-=-=-
# 4. Training.
#=-=-=-=-=-=-=-=-=-=-=-=-

mlp = NN()
mlp.Dense(50)
mlp.Dense(50)
#mlp.Dense(20) #Third hidden layer
#... add more layers
#with: mlp.Dense(number of hidden units)

mlp.fit(X_train, Y_train, y_train, params)

#=-=-=-=-=-=-=-=-=-=-=-=-
# 5. Evaluation.
#=-=-=-=-=-=-=-=-=-=-=-=-
predictions = mlp.predict(X_test)
accuracy = mlp.accuracy(X_test, Y_test, y_test)
print(accuracy)
