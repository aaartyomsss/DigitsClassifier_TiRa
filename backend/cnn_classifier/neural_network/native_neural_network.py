from django.conf import settings
import pathlib
import os
import numpy as np # Package that simplifies linear algebra
import pandas as pd # Allows to process better CSV data
import math

class NativeNeuralNetwork:

    def __init__(self, alpha, batch_size, training_size):
        self.y = None
        self.X = None

        self.test_y = None
        self.test_X = None

        self.alpha = alpha
        self.pixels_per_image = 784
        self.output_layer_size = 10
        self.batch_size = batch_size
        self.training_size = training_size
        self.kernel_rows = 3
        self.kernel_cols = 3
        self.num_kernels = 16
        self.hidden_size = ((28 - self.kernel_rows) * 
                            (28 - self.kernel_cols)) * self.num_kernels

        np.random.seed(1)


        # first matrix of weights that connects input layer with hidden layer of size 40
        self.weights_0_1 = 0.02*np.random.random((self.pixels_per_image, self.hidden_size)) - 0.01
        # Second matrix that connects hidden layer with output layer
        self.weights_1_2 =  0.2*np.random.random((self.hidden_size, self.output_layer_size)) - 0.1
        self.kernels = 0.02 * np.random.random((self.kernel_rows*self.kernel_cols, self.num_kernels)) - 0.01
    
    def train_network(self, iterations, preliminary_stop_after=15):
        # As an optimasation stop training when error gets bigger multiple times in a row
        # When throughout the training process we get that our error is getting greater
        # And does not reducer over certain amount of time - model is overfitted and there is no
        # need to continue
        minimal_error = math.inf
        times_error_greater_than_minimum = 0

        for z in range(iterations):
            error = 0.0
            correct = 0
            # We limit training data due to testing purposes
            # In order to increase the speed of learning
            for i in range(int(len(self.X[:self.training_size]) / self.batch_size)):
                batch_start, batch_end = (i * self.batch_size, (i + 1) * self.batch_size)
                layer_0 = self.X[batch_start:batch_end]
                # print("LAYER 0 SHAPE BEOFRE TRANSOFRMATION ", layer_0.shape)
                # Batch of images in format of 28x28
                # Reshaping is done in order to implement the 
                # "scanning" of the picture by using 3x3 filter
                # In other words layer zero has shape (batch_size, 28, 28)
                layer_0 = layer_0.reshape(layer_0.shape[0], 28, 28)
                
                # print("Shape of layer 0 ", layer_0.shape)
                # Implementation of convolutional layer
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

                dropout_mask = np.random.randint(2, size=layer_1.shape)
                layer_1 *= dropout_mask * 2
                
                layer_2 = self.softmax(np.dot(layer_1, self.weights_1_2))

                # Calculating how much our prediciton differs from the
                # Actual result
                error += np.sum((self.y[batch_start:batch_end] - layer_2) ** 2)

                for k in range(self.batch_size):
                    correct += int(np.argmax(layer_2[k:k + 1]) \
                                == np.argmax(self.y[batch_start + k:batch_end + k +1]))
                    # This block is the part of the training algorithm that
                    # is called backpropagation. More details can be found in
                    # documentation related to native implementation
                    layer_2_delta = ((self.y[batch_start:batch_end] - layer_2) 
                                      / (self.batch_size * layer_2.shape[0]))
                    layer_1_delta = layer_2_delta.dot(self.weights_1_2.T) * self.tanh2deriv(layer_1)

                    layer_1_delta *= dropout_mask

                    self.weights_1_2 += self.alpha * layer_1.T.dot(layer_2_delta)
                    layer_1_delta_reshape = layer_1_delta.reshape(kernel_output.shape)
                    kernel_update = flattened_input.T.dot(layer_1_delta_reshape)
                    self.kernels += self.alpha * kernel_update
            
            print(f"For iteration {z} was {error}")

            # if error < minimal_error:
            #     minimal_error = error
            #     times_error_greater_than_minimum = 0
            # else:
            #     times_error_greater_than_minimum += 1
            #     if times_error_greater_than_minimum == preliminary_stop_after:
            #         break

    # This is required for the convolutional layer. We will use this function
    # to select the part of the image.
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

    def flatten_input(self, sections):
        # Combine the sections of the images and flatten it 
        # to again use in CNN more conveniently

        # extended_inpit - format (200, 625, 3, 3)
        # In other words batch (200) of 625 picters (sections that we collected)
        # each of which has size of 3 x 3
        extended_input = np.concatenate(sections, axis=1)
        es = extended_input.shape
        return (extended_input.reshape(es[0]*es[1], -1), es)

    def test_network(self):
        correct = 0
        test_length = len(self.test_X)

        for i in range(test_length):
            layer_0 = self.test_X[i:i+1]
            layer_0 = layer_0.reshape(layer_0.shape[0], 28, 28)

            sections = self.get_sections(layer_0)
            flattened_input, es = self.flatten_input(sections)
            kernel_output = flattened_input.dot(self.kernels)
            layer_1 = self.tanh(kernel_output.reshape(es[0], -1))
            layer_2 = np.dot(layer_1, self.weights_1_2)

            correct += int(np.argmax(layer_2) ==
                           np.argmax(self.test_y[i:i+1]))


        return (correct / test_length) * 100

    def tanh(self, x):
        return np.tanh(x)

    def tanh2deriv(self, output):
        return 1 - (output ** 2)

    def softmax(self, x):
        temp = np.exp(x)
        return temp / np.sum(temp, axis=1, keepdims=True)

    def relu(self, x):
        # returns x in case x > 0 else 0. 
        # Possible due to the fact that true-false are can also be represented as 1 and 0
        return (x >= 0) * x 

    def relu2deriv(self, x):
        return x >= 0 
    
    def predict(self, image):
        relu = lambda x: (x >= 0) * x 
        layer_0 = image
        layer_1 = relu(np.dot(layer_0, self.weights_0_1))
        layer_2 = np.dot(layer_1, self.weights_1_2)
        return np.argmax(layer_2)
        
    def _get_saved_network_path(self, test_env=False):
        if test_env:
            return f'{settings.BASE_DIR}/native_network_test.npy'
        return f'{settings.BASE_DIR}/native_network.npy'

    def save_network(self, test_env=False):
        with open(self._get_saved_network_path(test_env), 'wb') as f:
            np.save(f, self.kernels)
            np.save(f, self.weights_1_2)
            

    def load_network(self, test_env=False):
        try: 
            with open(self._get_saved_network_path(test_env), 'rb') as f:
                self.kernels = np.load(f)
                self.weights_1_2 = np.load(f)
        except FileNotFoundError:
            self.load_data()
            self.train_network(300)
            self.save_network()

    def remove_saved_network(self, test_env=False):
        # /cnn_classifier/neural_network
        os.remove(self._get_saved_network_path(test_env))

    
    def load_data(self):
        path = pathlib.Path(__file__).parent.resolve()
        # The data provided is already in required format, aka pixels and their value
        # So we do not need to covert it
        # Basically copy-paste from the scikit implementation
        try:
            training_data = pd.read_csv(f'{path}/mnist_train.csv')
            test_data = pd.read_csv(f'{path}/mnist_test.csv')
        except:
            raise Exception("You do not have dataset downloaded")
        else:
            y_unformatted = training_data['label'].values.flatten()
            self.y = self.covert_labels(y_unformatted)
            self.X = training_data.drop(['label'], axis=1).values / 255

            self.test_y = self.covert_labels(test_data['label'].values.flatten())
            self.test_X = test_data.drop(['label'], axis=1).values / 255


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