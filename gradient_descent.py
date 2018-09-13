import helpers
from gradient_calculator import BackPropagationBasedCalculator
from data_source import BatchesIterator


class GradientDescent:
    """An implementation of gradient descent algorithm.
    
    It takes an instance NeuralNet as a parameter, mutates and trains it.
    
    Public methods:
        training_epoch: train for 1 epoch
        train: train for a given number of iterations (epochs)
    """
    class InvalidLearningRate(Exception):
        pass

    def __init__(self, neural_net, cost_function, learning_rate=3.0):
        """
        Set initial learning configuration.
        
        :param neural_net: an instance of NeuralNet class to train 
        :param cost_function: an instance of CostFunction sub class
        :param learning_rate: a float, determines the size of the update step
        """
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
        """
        Perform a single iteration of gradient descent.
        
        :param data_src: an instance of DataSource sub class 
        :return: None
        """
        cost_function = self._cost_function
        gradient_calculator = BackPropagationBasedCalculator(data_src=data_src,
                                                             neural_net=self._nnet,
                                                             cost_function=cost_function)
        wgrad, bgrad = gradient_calculator.compute_gradients()
        self.update_weights(weight_gradient=wgrad)
        self.update_biases(bias_gradient=bgrad)

    def train(self, data_src, nepochs):
        """Train a neural net for a given number of iterations.
        
        :param data_src: an instance of DataSource sub class
        :param nepochs: int, number of iterations to do
        """
        for i in range(nepochs):
            self.training_epoch(data_src=data_src)


class StochasticGradientDescent(GradientDescent):
    """A subclass of GradientDescent. It implements a stochastic gradient descent.
    
    Unlike GradientDescent instances, instances of this class will update
    parameters using a gradient calculated on a small subset of training set
    (say, 50 or 100 training examples). This allows for a much faster training.
    
    Overridden methods:
        training_epoch
    """
    class InvalidBatchSize(Exception):
        pass

    def __init__(self, neural_net, cost_function,
                 learning_rate=3.0, batch_size=50):
        """
        Initial configuration.
        
        :param neural_net: an instance of NeuralNet class to train 
        :param cost_function: an instance of CostFunction class
        :param learning_rate: float, determines how large is the update step
        :param batch_size: int, number of training examples used to compute a gradient
        """
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
