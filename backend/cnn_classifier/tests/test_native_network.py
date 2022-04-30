import pytest
from cnn_classifier.neural_network.native_neural_network import NativeNeuralNetwork
import time
import os

@pytest.fixture
def get_default_network():
    n = NativeNeuralNetwork(
        alpha=0.001,
        batch_size=200,
        training_size=14000,
        hidden_size=40
    )
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
    name = 'native_network.sav'
    n = get_default_network
    n.remove_saved_network(True)
    path = n._get_saved_network_path(True)
    assert os.path.exists(path) is False

# def test_network_training_speed_with_50_iterations(get_default_network):
#     n = get_default_network
#     start = time.perf_counter()
#     n.train_network(50)
#     finish = time.perf_counter()
#     assert finish - start < 900


# def test_network_accuracy_with_500_iters(get_default_network):
#     n = get_default_network
#     n.train_network(50)
#     accuracy = n.test_network()
#     assert accuracy >= 95


# def test_network_accuracy_with_100_iters(get_default_network):
#     n = get_default_network
#     n.train_network(100)
#     accuracy = n.test_network()
#     assert accuracy >= 95
