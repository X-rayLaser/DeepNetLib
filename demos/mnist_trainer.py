import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from neural_net import NetFactory
from data_source import PreloadSource, DataSetIterator
from gradient_descent import StochasticGradientDescent
from cost_functions import QuadraticCost, CrossEntropyCost
from activation_functions import Rectifier, Softmax
from datasets import mnist
import numpy as np


def estimate_accuracy(net, data_src):
    matches = 0
    m = data_src.number_of_examples()
    for x, y in DataSetIterator(data_src):
        a = net.feed(x)
        if np.argmax(a) == np.argmax(y):
            matches += 1

    return matches / float(m)


class MnistTrainer:
    def __init__(self, trainer_config):
        self._config = trainer_config
        self._data_src = None
        self._test_data_src = None
        self._neural_net = None
        self._cost_function = None
        self._load_mnist_examples()
        self._create_network()
        self._choose_cost_function()

    def _load_mnist_examples(self):
        mnist.download_dataset()

        train_data = mnist.get_training_data()
        test_data = mnist.get_test_data()

        dataset_size = self._config['dataset_size']
        if dataset_size:
            train_size = dataset_size
            test_size = dataset_size
        else:
            train_size = len(train_data)
            test_size = len(test_data)

        self._data_src = PreloadSource(train_data[:train_size])
        self._test_data_src = PreloadSource(test_data[:test_size])

    def _transform_examples(self):
        return 28, 28

    def _create_network(self):
        image_width, image_height = self._transform_examples()

        input_size = image_height * image_width
        output_size = 10
        hidden_sizes = self._config['hidden_layer_sizes']
        sizes = [input_size] + hidden_sizes + [output_size]
        self._neural_net = NetFactory.create_neural_net(
            sizes=sizes,
            hidden_layer_activation=self._config['hidden_activation'],
            output_layer_activation=self._config['output_activation']
        )

    def _choose_cost_function(self):
        reg_term = self._config['regularization_term']
        if self._config['loss_function'] == 'quadratic':
            loss_constructor = QuadraticCost
        else:
            loss_constructor = CrossEntropyCost

        self._cost_function = loss_constructor(neural_net=self._neural_net,
                                               l2_reg_term=reg_term)

    def start(self, nepoch=50, randomize=True):
        neural_net = self._neural_net
        if randomize:
            neural_net.randomize_parameters()

        sgd = StochasticGradientDescent(
            neural_net=neural_net, cost_function=self._cost_function,
            learning_rate=self._config['learning_rate'],
            batch_size=self._config['mini_batch_size']
        )

        for i in range(nepoch):
            sgd.training_epoch(data_src=self._data_src)
            print('{}th epoch passed:'.format(i))
            self._print_progress()
            print()

    def _print_progress(self):
        train_accuracy = estimate_accuracy(
            net=self._neural_net, data_src=self._data_src
        )
        test_accuracy = estimate_accuracy(
            net=self._neural_net, data_src=self._test_data_src
        )
        train_cost = self._cost_function.get_cost(data_src=self._data_src)
        test_cost = self._cost_function.get_cost(data_src=self._test_data_src)

        print('Accuracy on training and test data is '
              '{} % and {} % respectively'.format(train_accuracy * 100,
                                                  test_accuracy * 100)
              )

        print('Loss on train data is {}'.format(train_cost))
        print('Loss on test data is {}'.format(test_cost))


class TrainerConfig:
    @staticmethod
    def make_config(hidden_layer_sizes, hidden_activation=Rectifier,
                    output_activation=Softmax, loss_function=CrossEntropyCost,
                    L2_regularization_term=0, learning_rate=0.1,
                    mini_batch_size=10, dataset_size=None):
        return {
            'hidden_layer_sizes': hidden_layer_sizes,
            'hidden_activation': hidden_activation,
            'output_activation': output_activation,
            'loss_function': loss_function,
            'learning_rate': learning_rate,
            'regularization_term': L2_regularization_term,
            'mini_batch_size': mini_batch_size,
            'dataset_size': dataset_size
        }
