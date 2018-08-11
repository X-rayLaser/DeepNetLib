from random import shuffle
import helpers
from backprop import back_propagation


class GradientDescent:
    class InvalidLearningRate(Exception):
        pass

    def __init__(self, neural_net, learning_rate=3.0):
        if learning_rate <= 0:
            raise self.InvalidLearningRate('Learning rate must be a positive number')
        self._nnet= neural_net
        self._rate = learning_rate

    def update_weights(self, weight_gradient):
        weights = self._nnet.weights()
        for i in range(len(weights)):
            weights[i] -= self._rate * weight_gradient[i]

    def update_biases(self, bias_gradient):
        biases = self._nnet.biases()
        for i in range(len(biases)):
            biases[i] -= self._rate * bias_gradient[i]

    def training_epoch(self, examples):
        wgrad, bgrad = back_propagation(examples=examples, neural_net=self._nnet)
        self.update_weights(weight_gradient=wgrad)
        self.update_biases(bias_gradient=bgrad)

    def train(self, examples, nepochs):
        for i in range(nepochs):
            self.training_epoch(examples=examples)


class StochasticGradientDescent(GradientDescent):
    class InvalidBatchSize(Exception):
        pass

    def __init__(self, neural_net, learning_rate=0.1, batch_size=50):
        if not (type(batch_size) is int) or batch_size <= 0:
            raise self.InvalidBatchSize('Batch size must be a positive integer')
        GradientDescent.__init__(self, neural_net, learning_rate=learning_rate)
        self._batch_size = batch_size

    def shuffle_examples(self, examples):
        x_list, y_list = examples
        return helpers.shuffle_pairwise(x_list, y_list)

    def training_epoch(self, examples):
        x_list, y_list = examples
        x_batches = helpers.list_to_chunks(x_list,
                                           chunk_size=self._batch_size)
        y_batches = helpers.list_to_chunks(y_list,
                                           chunk_size=self._batch_size)
        batch_count = len(y_batches)
        for i in range(batch_count):
            mini_batch = (x_batches[i], y_batches[i])
            GradientDescent.training_epoch(self, examples=mini_batch)
