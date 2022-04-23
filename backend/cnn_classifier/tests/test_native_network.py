from cnn_classifier.neural_network.native_neural_network import NativeNeuralNetwork


def test_network_can_be_trained():
    # Simply to check if there were any erros in
    # matrix operations
    n = NativeNeuralNetwork()
    n.train_network(1)


def test_network_accuracy_with_50_iters():
    n = NativeNeuralNetwork()
    n.train_network(50)
    accuracy = n.test_network()
    assert accuracy >= 90


def test_network_accuracy_with_50_iters():
    n = NativeNeuralNetwork()
    n.train_network(100)
    accuracy = n.test_network()
    assert accuracy >= 90
