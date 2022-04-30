# Native CNN

As it was mentioned in a `README.md` file the implementation of the neural network was mainly done with the help of **Andrew W. Trask's** _*Grokking Deep Learning*_ book, which focuses on mainly on the implementation of the CNN rather than its underlying mathematical analysis.

## Current state of the the CNN

The neural network is currently rather simple. It does not use any complex algorithms. So far it is based on feeding a single image during the training process, comparing the output of network with the actual result and then passing back the this difference throughout the layer and tweaking the weights by this difference. In order to avoid linearity as an activation function - ReLu is used. Lastly, as a mean to deal with the overfitting - dropout mask is used on the hidden layer.

So far, the accuracy of the network has not been the worst, however in the future it will be optimised both by improving the recognition and the speed of the training.

### Gradient decent and backpropagation

> Reference: Neural Networks and Deep Learning by Michael Nielsen

In order for the neural network to learn, we have to find the most suitable coefficients for its weights. We begin, by evaluating how well our NN performs by taking the squared difference between our predicted output and the actual one by computing the `quadratic cost function` also known as `mean squared error (MSE)`

```python
 error += np.sum((self.y[batch_start:batch_end] - layer_2) ** 2)
```

The larger the value - the worse our network performs. Of course in an ideal world the value should be equal to 0. So the goal of the training algorithm is to minimize the cost function. In our neural network it is done with the help of `gradient decent`.

Well, why do we pick a gradient decent as an algorithm for finding the minima of the cost function? The answer lies in the fact that it is rather a simple method. We know that it is possible to find it by computing derivatives of the function and analysing them, but this task becomes intimidating when we deal with significant amount of variables... Which is usualy the case for neural networks.

How does then the algorithm of gradient decent work and can be visualised? If we limit ourselves to 3 dimensions and imagine some function at a certain random starting point then all the algorithm does is at each step (iteration) it moves into the direction of the minima of a function.

Let's have a look at the problem from the mathematical point, again considering 3 dimensions, where `z` is an output of a cost function and `x` & `y` are two variables affecting its value. Recall that we want to minize `z`, by changing the values of `x` and `y`. So it all boils down to the following:

<img height='100' src="https://render.githubusercontent.com/render/math?math=\Delta z \approx \frac{\partial z}{\partial x} \Delta x +
  \frac{\partial z}{\partial y} \Delta y">

So we are interested in finding out the values of `delta x` and `delta y`, such that `delta z` will be negative.

It is useful to rewrite this equation accordingly:
<img height='100' src="https://render.githubusercontent.com/render/math?math=\nabla z \equiv \left( \frac{\partial z}{\partial x}, 
  \frac{\partial z}{\partial y} \right)^T">

Being the gradient vector.

And we define the vector of changes:
<img height='100' src="https://render.githubusercontent.com/render/math?math=\Delta v \equiv (\Delta x, \Delta y)^T">

As a result whole equation can be rewritten as:
<img height='100' src="https://render.githubusercontent.com/render/math?math=\Delta z \approx \nabla z \cdot \Delta v">

From this equation we can also choose `delta v`, such that `delta z` will be negative. We suppose that:

<img height='100' src="https://render.githubusercontent.com/render/math?math=\Delta v = -\alpha \nabla C">

Where `alpha` is the learning rate (defined accordingly in our model). By plugging in this value into the equation above - it can be simply proven that `delta z` will always be negative. As a result, by applying those changes multiple amount of times - eventually we should arrive to the minima of a function. Following algorithm works for `N` amount of dimensions and procedure remains the same.

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
