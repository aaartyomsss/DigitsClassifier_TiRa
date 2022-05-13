from django.core.management.base import BaseCommand, CommandError
from cnn_classifier.neural_network.native_neural_network import NativeNeuralNetwork
import time

class Command(BaseCommand):
    help = 'Closes the specified poll for voting'

    def add_arguments(self, parser):
        parser.add_argument('training_size', type=int)
        parser.add_argument('iterations', type=int)

    def handle(self, *args, **options):
        iterations = options.get('iterations')
        training_size = options.get('training_size')
        if not iterations or not training_size:
            raise CommandError('Not all the arguements were provided')

        cnn = NativeNeuralNetwork(
            alpha=0.02,
            batch_size=300,
            training_size=training_size
        )
        cnn.load_data()
        start = time.perf_counter()
        cnn.train_network(iterations)
        end = time.perf_counter()
        accuracy = cnn.test_network()
        cnn.save_network()
        time_taken = end - start
        self.stdout.write(
            f"Network was trained with an accuracy of {accuracy} and it has taken {time_taken} seconds"
        )