from random import SystemRandom
import numpy as np
import math
from main import NeuralNet, NetFactory
import helpers
import gradient_descent
import cost_functions
from digit_drawing import DigitGenerator
from activation_functions import Rectifier, Softmax
from gradient_calculator import NumericalCalculator, BackPropagationBasedCalculator
from datasets import mnist
from data_source import PreloadSource
from data_source import DataSetIterator


def squared_sin_data_set():
    def f(x):
        return math.sin(x) ** 2

    return PreloadSource(
        helpers.generate_data(f=f, start_value=0, end_value=3.14, step_value=0.1)
    )


@when('I generate a data set from a function "sin(x)^2"')
def step(context):
    context.training_data = squared_sin_data_set()


@when('I remember initial cost value for that data set')
def step(context):
    data = context.training_data
    context.initial_cost = context.cost_function.get_cost(data)


@when('I train neural network on that data set for {nepoch} epochs')
def step(context, nepoch):
    data = context.training_data
    algo = context.learning_algorithm
    algo.train(nepochs=int(nepoch), data_src=data)


@then('the cost function gives much smaller error value than before')
def step(context):
    data = context.training_data
    latest_cost = context.cost_function.get_cost(data)
    assert context.initial_cost > latest_cost,\
        'Cost is still too large. Was {}, now {}'.format(context.initial_cost, latest_cost)


@when('I initialize simple neural net with default parameters')
def step(context):
    context.nnet = NetFactory.create_neural_net(sizes=[1, 10, 1])
    context.nnet.randomize_parameters()


@when('I choose to use gradient descent as learning algorithm')
def step(context):
    net = context.nnet
    context.learning_algorithm = gradient_descent.GradientDescent(
        neural_net=net, cost_function=context.cost_function
    )


@when('I train neural network to approximate a function "sin(x)^2"')
def step(context):
    examples = squared_sin_data_set()
    context.nnet.train(examples=examples)


@then('I feed it a set of inputs and get correct results')
def step(context):
    xs = [-10, -5, -1, 0, 1, 5, 10]
    for x in xs:
        y = math.sin(x)**2
        a = context.nnet.feed(np.array([x], float))
        assert abs(a[0] - y) < 0.1, 'a[0]={}, y={}'.format(a[0], y)


@when("I randomly initialize net's parameters")
def step(context):
    context.nnet.randomize_parameters()


@when('I compute the gradient for weights and biases by running back propagation')
def step(context):
    cost_func = context.cost_function
    calculator = BackPropagationBasedCalculator(
        data_src=context.training_data,
        neural_net=context.nnet,
        cost_function=cost_func
    )
    context.back_prop_gradients = calculator.compute_gradients()


@when('I compute the gradient for weights and biases by taking numerical derivatives')
def step(context):
    cost_func = context.cost_function
    calculator = NumericalCalculator(
        data_src=context.training_data,
        neural_net=context.nnet,
        cost_function=cost_func
    )
    context.numerical_gradients = calculator.compute_gradients()


@then('these two sets of gradients are the same')
def step(context):
    wgrad1, bgrad1 = context.back_prop_gradients
    wgrad2, bgrad2 = context.numerical_gradients
    assert helpers.gradients_equal(wgrad1, wgrad2)
    assert helpers.gradients_equal(bgrad1, bgrad2)


def examples_drawn_from_distributions(number_of_examples, seq_len, create_example):
    inputs = []
    outputs = []
    for n in range(number_of_examples):
        x, y = create_example(seq_len)
        inputs.append(x)
        outputs.append(y)

    return (inputs, outputs)


def create_example(slen):
    import random
    sr = random.SystemRandom()

    num_of_distrib = 3

    rv = sr.randint(0, num_of_distrib - 1)
    y = np.zeros((num_of_distrib,))
    y[rv] = 1.0

    if rv == 0:
        x = np.random.rand(slen)
    elif rv == 1:
        x = np.random.randn(slen)
    elif rv == 2:
        x = np.random.geometric(p=0.25, size=slen)
    return x, y


@when('I generate a data set consisting of sequences of length {seq_len} drawn from different distributions')
def step(context, seq_len):
    slen = int(seq_len)

    training_data = examples_drawn_from_distributions(
        number_of_examples=200, seq_len=slen, create_example=create_example
    )
    context.training_data = PreloadSource(training_data)

    test_data = examples_drawn_from_distributions(
        number_of_examples=50, seq_len=slen, create_example=create_example
    )
    context.test_data = PreloadSource(test_data)


@when('I initialize a neural net for binary classification with sizes {sizes_csv}')
def step(context, sizes_csv):
    sizes = [int(sz) for sz in sizes_csv.split(',')]
    context.nnet = NetFactory.create_neural_net(sizes=sizes)


@then('neural net gives less than {classification_error}% classification error on test data set')
def step(context, classification_error):
    required_accuracy = 100 - int(classification_error)

    num_of_examples = context.test_data.number_of_examples()
    matches = 0
    for x, y in DataSetIterator(context.test_data):
        a = context.nnet.feed(x)
        index = np.argmax(a)
        expected_index = np.argmax(y)
        if index == expected_index:
            matches += 1

    accuracy = float(matches) / num_of_examples * 100
    print('accuracy:', accuracy)
    assert accuracy >= required_accuracy


@when('I choose stochastic gradient descent as a learning algorithm with learning rate {rate}')
def step(context, rate):
    cost_function = context.cost_function
    r = float(rate)
    sgd = gradient_descent.StochasticGradientDescent(
        context.nnet, cost_function=cost_function, learning_rate=r
    )
    context.learning_algorithm = sgd


@when('I choose quadratic cost function')
def step(context):
    context.cost_function = cost_functions.QuadraticCost(
        neural_net=context.nnet
    )


@when('I choose cross entropy cost function')
def step(context):
    context.cost_function = cost_functions.CrossEntropyCost(
        neural_net=context.nnet
    )


@when('I create a training and testing data from MNIST data set')
def step(context):
    mnist.download_dataset()
    context.training_data = PreloadSource(mnist.get_training_data())
    context.test_data = PreloadSource(mnist.get_test_data())


@when('I train a digit generator')
def step(context):
    mnist.download_dataset()
    pixels_to_categories = mnist.get_training_data()
    generator = DigitGenerator()
    generator.train(pixels_to_categories=pixels_to_categories)
    context.generator = generator


@then('out of {n} generated digits {accuracy}% or more are indeed digits')
def step(context, n, accuracy):
    sr = SystemRandom()

    matches = 0
    for i in range(int(n)):
        digit = sr.randint(0, 9)
        pixels = context.generator.generate_digit(digit)
        output = context.nnet.feed(x=pixels)
        index = np.argmax(output)
        if index == digit:
            matches += 1

    print(matches / float(n) * 100)
    assert matches / float(n) * 100 >= int(accuracy)


@when('I memorize weights and biases')
def step(context):
    context.weights = context.nnet.weights()
    context.biases = context.nnet.biases()


@when('I save neural net weights and biases to file "{fname}"')
def step(context, fname):
    context.nnet.save(fname)


@when('I initialize neural net parameters from a file "{fname}"')
def step(context, fname):
    context.new_net = NeuralNet.create_from_file(fname)


@then('new net parameters match the parameters of old neural net')
def step(context):
    num_of_layers = len(context.new_net.weights())
    assert num_of_layers == len(context.nnet.weights())
    assert len(context.new_net.biases()) == len(context.nnet.biases())

    for i in range(num_of_layers):
        assert np.allclose(context.new_net.weights()[i], context.nnet.weights()[i])
        assert np.allclose(context.new_net.biases()[i], context.nnet.biases()[i])


@when('I choose rectifier activation function')
def step(context):
    for layer in context.nnet.layers():
        layer.set_activation(Rectifier)


@when('I choose rectifier activation function for hidden layer(s)')
def step(context):
    layers = context.nnet.layers()
    for i in range(len(layers) - 1):
        layers[i].set_activation(Rectifier)


@when('I choose softmax activation function for output layer')
def step(context):
    context.nnet.layers()[-1].set_activation(Softmax)
