import helpers
from gradient_calculator import BackPropagationBasedCalculator
from data_source import BatchesIterator


class GradientDescent:
    class InvalidLearningRate(Exception):
        pass

    def __init__(self, neural_net, cost_function, learning_rate=3.0):
        if learning_rate <= 0:
            raise self.InvalidLearningRate('Learning rate must be a positive number')
        self._nnet= neural_net
        self._rate = learning_rate
        self._cost_function = cost_function

    def update_weights(self, weight_gradient):
        weights = self._nnet.weights()
        for i in range(len(weights)):
            weights[i] -= self._rate * weight_gradient[i]

    def update_biases(self, bias_gradient):
        biases = self._nnet.biases()
        for i in range(len(biases)):
            biases[i] -= self._rate * bias_gradient[i]

    def training_epoch(self, data_src):
        cost_function = self._cost_function
        gradient_calculator = BackPropagationBasedCalculator(data_src=data_src,
                                                             neural_net=self._nnet,
                                                             cost_function=cost_function)
        wgrad, bgrad = gradient_calculator.compute_gradients()
        self.update_weights(weight_gradient=wgrad)
        self.update_biases(bias_gradient=bgrad)

    def train(self, data_src, nepochs):
        for i in range(nepochs):
            self.training_epoch(data_src=data_src)


class StochasticGradientDescent(GradientDescent):
    class InvalidBatchSize(Exception):
        pass

    def __init__(self, neural_net, cost_function,
                 learning_rate=3.0, batch_size=50):
        if not (type(batch_size) is int) or batch_size <= 0:
            raise self.InvalidBatchSize('Batch size must be a positive integer')
        GradientDescent.__init__(self, neural_net, cost_function,
                                 learning_rate=learning_rate)
        self._batch_size = batch_size

    def shuffle_examples(self, examples):
        x_list, y_list = examples
        return helpers.shuffle_pairwise(x_list, y_list)

    def training_epoch(self, data_src):
        for batch_src in BatchesIterator(data_src, batch_size=self._batch_size):
            GradientDescent.training_epoch(self, data_src=batch_src)
