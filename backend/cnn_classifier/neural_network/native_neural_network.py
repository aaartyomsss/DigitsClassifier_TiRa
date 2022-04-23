from cProfile import label
import pathlib
import os
import numpy as np # Package that simplifies linear algebra
import pandas as pd # Allows to process better CSV data

class NativeNeuralNetwork:

    def __init__(self):
        path = pathlib.Path(__file__).parent.resolve()
        # The data provided is already in required format, aka pixels and their value
        # So we do not need to covert it
        # Basically copy-paste from the scikit implementation
        training_data = pd.read_csv(f'{path}/mnist_train.csv')
        test_data = pd.read_csv(f'{path}/mnist_test.csv')

        y_unformatted = training_data['label'].values.flatten()
        # TODO: convertation should take place else where
        self.y = self.covert_labels(y_unformatted)
        self.X = training_data.drop(['label'], axis=1).values / 255

        # TODO: convertation should take place else where
        self.test_y = self.covert_labels(test_data['label'].values.flatten())
        self.test_X = test_data.drop(['label'], axis=1).values / 255

        # Next we hardcore some params that will later be initilized via constructor
        self.alpha = 0.015
        self.hidden_size = 40
        self.pixels_per_image = 784
        self.output_layer_size = 10
        np.random.seed(1)

        # first matrix of weights that connects input layer with hidden layer of size 40
        self.weights_0_1 = 0.2*np.random.random((self.pixels_per_image, self.hidden_size)) - 0.1
        # Second matrix that connects hidden layer with output layer
        self.weights_1_2 =  0.2*np.random.random((self.hidden_size, self.output_layer_size)) - 0.1
    
    def train_network(self, iterations):
        # returns x in case x > 0 else 0. 
        # Possible due to the fact that true-false are can also be represented as 1 and 0 
        relu = lambda x: (x >= 0) * x 
        relu2deriv = lambda x: x >= 0 # 1 in case x > 0 else 0

        for _ in range(iterations):
            error = 0.0
            correct = 0
            # TODO: we limit training data due to testing purposes
            # In order to increase the speed of learning
            for i in range(len(self.X[:1000])):
                layer_0 = self.X[i:i+1]
                # We apply relu to set negative weights to zero in 
                # order to avoid linearity between layers
                layer_1 = relu(np.dot(layer_0, self.weights_0_1))

                dropout_mask = np.random.randint(2, size=layer_1.shape)
                layer_1 *= dropout_mask * 2
                
                layer_2 = np.dot(layer_1, self.weights_1_2)

                correct += int(np.argmax(layer_2) == np.argmax(self.y[i:i+1]))

                # Calculating how much our prediciton differs from the
                # Actual result
                error += np.sum((self.y[i:i+1] - layer_2) ** 2)

                # This block is the part of the training algorithm that
                # is called backpropagation. More details can be found in
                # documentation related to native implementation
                layer_2_delta = self.y[i:i+1] - layer_2
                layer_1_delta = layer_2_delta.dot(self.weights_1_2.T) * relu2deriv(layer_1)

                layer_1_delta *= dropout_mask

                self.weights_1_2 += self.alpha * layer_1.T.dot(layer_2_delta)
                self.weights_0_1 += self.alpha * layer_0.T.dot(layer_1_delta)


    def test_network(self):
        correct = 0
        test_length = len(self.test_X)
        relu = lambda x: (x >= 0) * x 

        for i in range(test_length):
            layer_0 = self.test_X[i:i+1]
            layer_1 = relu(np.dot(layer_0, self.weights_0_1))
            layer_2 = np.dot(layer_1, self.weights_1_2)
            
            # np.argmax finds index of max value in array
            # if they are equal - prediction is correct
            correct += int(np.argmax(layer_2) == np.argmax(self.test_y[i:i+1]))
        return (correct / test_length) * 100

    
    def predict(self, image):
        relu = lambda x: (x >= 0) * x 
        layer_0 = image
        layer_1 = relu(np.dot(layer_0, self.weights_0_1))
        layer_2 = np.dot(layer_1, self.weights_1_2)
        return np.argmax(layer_2)
        
    def covert_labels(self, y):
        # As the output layer of our neural network will have 10 nodes
        # Each of which will represent certain probability => original
        # format of y is not suitable for teaching our neural network.
        # Thus, every label has to be converted from integer value into
        # an array with probability one at certain index. for example:
        # 5 => [0 0 0 0 0 1 0 0 0 0] is the way in which five will be converted
        test_labels = np.zeros((len(y), 10))
        for i, n in enumerate(y):
            test_labels[i][n] = 1
        return test_labels