import pathlib
import os
import numpy as np # Package that simplifies linear algebra
import pandas as pd # Allows to process better CSV data
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import pickle


class NeuralNetwork:

    def __init__(self):
        path = pathlib.Path(__file__).parent.resolve()
        # The data provided is already in required format, aka pixels and their value
        # So we do not need to covert it
        training_data = pd.read_csv(f'{path}/mnist_train.csv')
        self.test_data = pd.read_csv(f'{path}/mnist_test.csv')
        self.y = training_data['label'].values.flatten()
        self.X = training_data.drop(['label'], axis=1).values
        self.classifier = None

    def train_model(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, random_state=42)

        clf = MLPClassifier(solver='lbfgs', random_state=1, hidden_layer_sizes=(100, ))
        clf.fit(X_train/255.0, y_train)
        self.X_test = X_test
        self.y_test = y_test
        self.classifier = clf

    def test_model(self):
        pred = self.classifier.predict(self.X_test)
        return np.mean(pred == self.y_test)

    def save_trained_model(self, testing=False):
        # This will be later substituted with the database implementation
        name = 'trained_network.sav' if not testing else 'test_trained_network.sav'
        pickle.dump(self.classifier, open(name, 'wb'))

    def get_root_directory(self):
        # Getting root directory i.e. /backend
        path = pathlib.Path(__file__).parent.parent.parent.resolve()
        return str(path)

    def remove_model(self, testing=False):
        name = 'trained_network.sav' if not testing else 'test_trained_network.sav'
        path = self.get_root_directory()
        os.remove(f'{path}/{name}')


    def open_trained_model(self, testing=False):  
        name = 'trained_network.sav' if not testing else 'test_trained_network.sav'
        root_dir = self.get_root_directory()
        try:
            loaded_model = pickle.load(open(f'{root_dir}/{name}', 'rb'))
            return loaded_model
        except:
            raise Exception('Model file was not found')