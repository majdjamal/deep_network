# Neural Network

## Overview

Multilayer Perceptron with k Hidden Layers, built from scratch using NumPy. You can choose the number of hidden layers with the .Dense() method, a demonstration is found in experiments.py.

Hidden Layer values are computed with ReLU (eq. 1) and output probabilities with Softmax (eq. 2). The architecture is demonstrated in Figure 1.

<img src="https://i.ibb.co/fXNnNbn/Ska-rmavbild-2021-04-12-kl-21-31-21.png" width="175" height="35">


<img src="https://i.ibb.co/ZG6ghz8/Ska-rmavbild-2021-04-12-kl-21-32-55.png" width="190" height="55">


<img src="https://i.ibb.co/D4cX5Xm/layers.png" width="410" height="250">
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
