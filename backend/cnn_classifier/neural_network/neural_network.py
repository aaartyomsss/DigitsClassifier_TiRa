import pathlib
import numpy as np # Package that simplifies linear algebra
import pandas as pd # Allows to process better CSV data
import pathlib
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report


class NeuralNetwok:

    def __init__(self):
        path = pathlib.Path(__file__).parent.resolve()
        # The data provided is already in required format, aka pixels and their value
        # So we do not need to covert it
        training_data = pd.read_csv(f'{path}/mnist_train.csv')
        test_data = pd.read_csv(f'{path}/mnist_test.csv')
        y = training_data['label'].values.flatten()
        X = training_data.drop(['label'], axis=1).values
        y_test = None
        classifier = None
        X_test = None


    def train_model(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, random_state=42)

        clf = MLPClassifier(solver='lbfgs', random_state=1, hidden_layer_sizes=(100, ))
        model = clf.fit(X_train/255.0, y_train)
        self.X_test = X_test
        self.y_test = y_test
        self.classifier = clf

    def test_model(self):
        pred = self.classifier.predict(self.X_test)
        return np.mean(pred == self.y_test)
