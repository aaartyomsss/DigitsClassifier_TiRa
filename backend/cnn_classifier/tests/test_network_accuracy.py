from matplotlib import testing
from cnn_classifier.neural_network.neural_network import NeuralNetwork
import time
import os


def test_cnn_training_speed():
    # Realistically, we should not be concerned about the training speed too much
    # As re-evaluation of the weights and biases should not happen often.
    # Only after significant amount of new data has been added.
    # However, the speed of training can really have an affect when we are trying
    # to optimise and tweak the parameters of the CNN, which can be quite time consuming 
    # TODO: Aff Coverage.py to track test coverage
    nn = NeuralNetwork()
    start = time.perf_counter()
    nn.train_model()
    finish = time.perf_counter()
    assert finish - start < 120


def test_cnn_accuracy():
    nn = NeuralNetwork()
    nn.train_model()
    assert nn.test_model() > 0.9
    assert nn.test_model() > 0.95


def test_cnn_is_saved():
    nn = NeuralNetwork()
    nn.train_model()
    nn.save_trained_model(True)
    model = nn.open_trained_model(True)
    assert model is not None

def test_cnn_model_removed():
    name = 'test_trained_network.sav'
    nn = NeuralNetwork()
    path = nn.get_root_directory()
    nn.remove_model(testing=True)
    assert os.path.exists(f'{path}/{name}') is False