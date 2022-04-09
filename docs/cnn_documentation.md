# Prinicple of CNN workflow

## Firstly, images should be converted to the understandable format... For the machine

The way computers operates with images differs quite a bit from how humans do. In order for it to "understand" them - the should be represented numerically. In our case, as we are dealing with black and white images - the case is rather simple. Will have an array of 28 x 28 = 784 elements each of which would have a value from 0 to 255. Closer to 0 - the darker the pixel. Closer to 255 - the lighter it is. If we would have had RGB format, then we would have had 3 array of 784 element representing different tones of red, green and blue colors.

In order to convert the uploaded image made in `UploadImageView`, we should do the following:

```python
    from PIL import Image
    from numpy import asarray

    image = Image.open('<file_name.png>').convert('L')
    data = array(image)
```

What by default any image is opened in a RGBA format and in order to get it to greyscale format we use `.convert('L')`

## Dataset that the CNN is trained on

The MNIST dataset consists of 60000 labeled (training) examples and 10000 images used for testing. Each image is in format of 28 x 28 pixels, with black background and white handwritten digit on it. The format of data is already well formatted for our purposes - it is a CSV file with numeric representation of pixels. Thus for us in order to train the model it simply boils down to opening the files and then, we separating the label and actual pixels as y and X:

```python
training_data = pd.read_csv(f'{path}/mnist_train.csv')
self.y = training_data['label'].values.flatten()
self.X = training_data.drop(['label'], axis=1).values
```

Lastly, we split the data for testing/training purposes, specify the model (in our case multilayer perceptron) and 'feed' the data to it.

```python
 X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, random_state=42)

clf = MLPClassifier(solver='lbfgs', random_state=1, hidden_layer_sizes=(100, ))
clf.fit(X_train/255.0, y_train) ## Scaling the pixel values to range from 0 to 1
```

## Prediction

After the model has been trained, prediciting is rather simple. We just have to pass the array of arrays (numerically represented images) and as a result we will receive and array of predicted values.

```python
img = Image.open(image_file).convert('L')
# Numeric representation of the image
data = array(img)
# As our CNN was trained using this format -
# Input date should be formatted accordingly
data = data.reshape(784, )

cnn = NeuralNetwork()
cnn = cnn.open_trained_model()

## Neural network accepts array of data and returns array of predictions
result = cnn.predict([data])
# Sending back the predicted result
return Response(result[0])
```

## Saving model

As the evaluation of weights and biases is quite time consuming - the model after the initial training should be saved and loaded when it is required to evalute some values. Those methods could be found in `neural_network/neural_network.py` and using them is rather straightforward.
