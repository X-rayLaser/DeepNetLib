from main import NeuralNet, back_propagation
import helpers
import numpy as np
import math


def squared_sin_data_set():
    def f(x):
        return math.sin(x) ** 2

    return helpers.generate_data(f=f, start_value=0, end_value=3.14, step_value=0.1)


@when('I generate a data set from a function "sin(x)^2"')
def step(context):
    context.training_data = squared_sin_data_set()


@when('I remember initial cost value for that data set')
def step(context):
    data = context.training_data
    context.initial_cost = context.nnet.get_cost(data)


@when('I train neural network on that data set for {nepoch} epochs')
def step(context, nepoch):
    data = context.training_data
    context.nnet.train(nepochs=int(nepoch), examples=data)


@then('the cost function gives much smaller error value than before')
def step(context):
    data = context.training_data
    latest_cost = context.nnet.get_cost(data)
    assert context.initial_cost > 5 * context.nnet.get_cost(data),\
        'Cost is still too large. Was {}, now {}'.format(context.initial_cost, latest_cost)


@when('I initialize simple neural net with default parameters')
def step(context):
    context.nnet = NeuralNet(layer_sizes=[1, 10, 1])
    context.nnet.randomize_parameters()


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
    context.back_prop_gradients = back_propagation(examples=context.training_data,
                                                   neural_net=context.nnet)


@when('I compute the gradient for weights and biases by taking numerical derivatives')
def step(context):
    context.numerical_gradients = helpers.back_propagation_slow(
        examples=context.training_data, neural_net=context.nnet
    )


@then('these two sets of gradients are the same')
def step(context):
    wgrad1, bgrad1 = context.back_prop_gradients
    wgrad2, bgrad2 = context.numerical_gradients
    assert helpers.gradients_equal(wgrad1, wgrad2)
    assert helpers.gradients_equal(bgrad1, bgrad2)
