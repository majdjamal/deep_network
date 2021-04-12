# Neural Network

## Overview

Multilayer Perceptron with k Hidden Layers, built from scratch using NumPy. You can choose the number of hidden layers with the .Dense() method, a demonstration is found in experiments.py. 

Hidden Layer values are computed with ReLU (eq. 1) and output probabilities with Softmax (eq. 2). The architecture is demonstrated in Figure 1. 

<img src="https://i.ibb.co/GWTZhdS/relu.png" width="100" height="50">
[eq. 1]

<img src="https://i.ibb.co/DgDPnQQ/Ska-rmavbild-2021-04-12-kl-21-23-35.png" width="100" height="50">
[eq. 2]

<img src="https://i.ibb.co/D4cX5Xm/layers.png" width="375" height="225">
(Figure 1. Architecture of a k-layer Neural Network)

## Setup 

### Data
Required shape of the data is X.shape = (Ndim, Npts). Labels can have shape y.shape = (Npts,)

### Environment
This model requires Python 3 and packages NumPy and Matplotlib.pyplot. 

### Demonstration 
Navigate to the repository in your terminal and type:

```bash
python experiment.py
```

### Questions
Feel free to send me your questions to Majdj@kth.se.
