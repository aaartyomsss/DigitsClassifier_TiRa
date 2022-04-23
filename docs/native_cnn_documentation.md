# Native CNN

As it was mentioned in a `README.md` file the implementation of the neural network was mainly done with the help of **Andrew W. Trask's** _*Grokking Deep Learning*_ book, which focuses on mainly on the implementation of the CNN rather than its underlying mathematical analysis.

## Current state of the the CNN

The neural network is currently rather simple. It does not use any complex algorithms. So far it is based on feeding a single image during the training process, comparing the output of network with the actual result and then passing back the this difference throughout the layer and tweaking the weights by this difference. In order to avoid linearity as an activation function - ReLu is used. Lastly, as a mean to deal with the overfitting - dropout mask is used on the hidden layer.

So far, the accuracy of the network has not been the worst, however in the future it will be optimised both by improving the recognition and the speed of the training.

### Backpropagation

-----todo-----

### Regularization

One of the main challenges of building a CNN is to avoid overfitting. In simple terms - overfitting is a state when our model has learned to recognize ideally training data, however because of that it won't be able to recognize unknown features quite well.

In order to avoid the overfitting, we will be using one of the simples, yet effective techniques - dropout. What it does is that randomly sets certain nodes to 0. Why so? Well, yet again in simple terms, following algorithm will allow to imitate neural networks of a smaller size, which are less likely to be affected by overfitting due to the fact that they for example are unable to capture noise (i.e. small details within our data).

The dropout mask in the code is done accordingly:

```python

...

dropout_mask = np.random.randint(2, size=layer_1.shape)
layer_1 *= dropout_mask * 2

...

layer_1_delta *= dropout_mask

```

The `dropout_mask` creates a matrix of the same shape as layer 1 consisting 50% of zeros and other 50% of ones, simulating the "turning off" mechanism for the node. We multiply then the product by 2 due to the fact that the weighted sum in the layer 2 will be less by a half, because of the dropout. However, when we actually use the trained network - all the neurons are working and weighted sum will be back to normal. In order to avoid this inequality - multiplication is performed.

`layer_1_delta *= dropout_mask` is perfomed due to the obvious reasons of the way backpropagation algorithm is performed.
