import pytest
from cnn_classifier.neural_network.native_neural_network import NativeNeuralNetwork
import time
import os

@pytest.fixture
def get_default_network():
    n = NativeNeuralNetwork(
        alpha=0.2,
        batch_size=200,
        training_size=1000,
    )
    n.load_data()
    return n


def test_network_can_be_trained(get_default_network):
    # Simply to check if there were any erros in
    # matrix operations
    n = get_default_network
    n.train_network(1)

def test_network_can_be_saved(get_default_network):
    n = get_default_network
    n.save_network(True)

def test_network_can_be_opened(get_default_network):
    n = get_default_network
    n.load_network(True)

def test_network_can_be_removed(get_default_network):
    n = get_default_network
    n.remove_saved_network(True)
    path = n._get_saved_network_path(True)
    assert os.path.exists(path) is False

def test_network_training_speed_with_1_iterations(get_default_network):
    print("speed test")
    n = get_default_network
    start = time.perf_counter()
    n.train_network(1)
    finish = time.perf_counter()
    print(finish - start)
    assert finish - start < 60

def test_network_accuracy_with_300_iters(get_default_network):
    n = get_default_network
    n.train_network(300)
    accuracy = n.test_network()
    print(accuracy)
    assert accuracy >= 90
