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

## Convolutional layer - tool for the pattern detection

filters detect patterns
what is a pattern? edges? curves? in general geometric filters (simple ones when amount of layers is low)
3 x 3 matrix that goes over each 3 x 3 of our image - dot product evaluation

Convolution layer is a tool that is used to detect certain patterns in images. Usually, the bigger the amount of layer - the more complex patterns become.
In order to help to achieve this effect CNNs use filters (i.e. kernels, convolutional matrix or masks) to help recognize patters. There are different approaches on which kernels to use.
There are certain matrices that help recognize edges, curves and other shapes. Other reduce the noise of an image or sharpen it.

> NB! Our CNN does not use any additonal kernels. However, that could have helped improving the accuracy of CNN

In general the process of convolution works as follows. We "scan" an image using ceratain amount of pixel and get sections of an image.

```python
def get_image_section(self, layer, row_from, row_to, col_from, col_to):
    sub_section = layer[:, row_from:row_to, col_from:col_to]
    # -1 leaves for the numpy to detect the shape of the image
    # so that the end shape would have been compatible with the original one
    return sub_section.reshape(-1, 1, row_to-row_from, col_to-col_from)

def get_sections(self, layer):
    sections_of_images = []
    for row_start in range(layer.shape[1] - self.kernel_rows):
        for col_start in range(layer.shape[2] - self.kernel_cols):
            section = self.get_image_section(layer,
                                              row_start,
                                              row_start+self.kernel_rows,
                                              col_start,
                                              col_start+self.kernel_cols)
            sections_of_images.append(section)
    return sections_of_images
```

After getting those sections we can imagine them being independent images based on which we will make certain predictions in the next layer. Implementation in the code:

```python
sections_of_images = self.get_sections(layer_0)

# Combine the sections of the images and flatten it
# to again use in CNN more conveniently
# flattened input is of format (125000, 9) which basically means
# that all 3 by 3 sections of picters from a batch are
# conveniently transformed into a matrix
flattened_input, es = self.flatten_input(sections=sections_of_images)
# print("FLATTENED INPUT SHAPE ", flattened_input.shape)

kernel_output = flattened_input.dot(self.kernels)
# print("KERNEL OUTP SHAPE ", kernel_output.shape)
# print("kernel reshape ", kernel_output.reshape(es[0], -1).shape)
# print("Hidden size ", self.hidden_size)

layer_1 = self.tanh(kernel_output.reshape(es[0], -1))
```

> Code might not be the cleanest and most understandable, due to a lot of formatting (reshaping) of the matrices

And in simple terms that is all that Convolutional layer that. It selects a part of image and makes a single prediction based on this part and passed it to the next layer.

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

## Accuracy report

Testing of the NN data was performed on a limited amount of data as otherwise testing would have taken too much time. Thus, it was chosen to use following base parameters

```python
training_size=1000,
iterations=300
```

Other parameters, such as: alpha coefficient, activation functions, hidden layer size and etc. were changing throughout the testing to find the best suitable combination of parameters that would lead to the greatest accuracy.

Testing case 1:

```python
# Parameters
hidden_layer=40,
alpha=0.001,
batch_size=200

## Activation function that was used: relu

## Accuracy after 300 iterations: 79.24%
```

Testing case 2:

```python
## NB! FALTY TEST BECAUSE OF THE ERRORS IN CODE
# Parameters
alpha=0.02,
batch_size=100,
hidden_size=100

## Activation function that was used: tanh on hidden layer and softmax on output layer

## Accuracy after 300 iterations: 73.77%
```

Testing case 3:

```python
## NB! FALTY TEST BECAUSE OF THE ERRORS IN CODE
# Parameters
alpha=0.0015,
batch_size=200,
hidden_size=100

## Activation function that was used: tanh on hidden layer and softmax on output layer

## Accuracy after 300 iterations: 48.77%
''' NB! Reason for such a low accuracy is the fact that alpha coefficient was too low
    This could have this could have somewhat been useful on whole 60k images, however
    it is still possible that the model would have been overtrained '''
```

Testing case 4:

```python
# Parameters
alpha=0.025,
batch_size=200,
hidden_size=100

## Activation function that was used: tanh on hidden layer and softmax on output layer

## Accuracy after 300 iterations: 86.16%
```

Testing case 5:

```python
## NB! FALTY TEST BECAUSE OF THE ERRORS IN CODE
# Parameters
alpha=0.04,
batch_size=200,
hidden_size=100

## Activation function that was used: tanh on hidden layer and softmax on output layer

## Accuracy after 300 iterations: 73.5%
```

Testing case 6:

```python
## NB! This test case differs a lot as convolutional layer was used
## And in my implementation the hidden layer is omitted and parameters for kernels
## are hardcoded.
# Parameters
alpha=0.2,
batch_size=200,
training_size=1000,

## Activation function that was used: tanh on hidden layer and softmax on output layer

## Accuracy after 300 iterations: 85.43%
```

Training model

```python
## Amount of images = 10000 and 50 iterations
cnn = NativeNeuralNetwork(
            alpha=0.02,
            batch_size=300,
            training_size=training_size
      )


## Result: Network was trained with an accuracy of 90.7 and it has taken 19688.528750200003 seconds
```
