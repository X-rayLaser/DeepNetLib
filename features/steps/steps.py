from main import NeuralNet
import numpy as np
import math


@when('I generate a data set from a function "sin(x)^2"')
def step(context):
    x = []
    y = []

    val_from = -100
    val_to = 100
    incr = 0.1
    farg = val_from
    while farg < val_to:
        v = np.array((1,), float)
        v[0] = math.sin(farg) ** 2
        x.append(farg)
        y.append(v)
        farg += incr
    context.training_data = (x, y)


@when('I remember initial cost value for that data set')
def step(context):
    data = context.training_data
    context.initial_cost = context.nnet.get_cost(data)


@when('I train neural network on that data set for {nepoch} epochs')
def step(context, nepoch):
    data = context.training_data
    context.nnet.train(training_epochs=nepoch, examples=data)


@then('the cost function gives much smaller error value than before')
def step(context):
    data = context.training_data
    latest_cost = context.nnet.get_cost(data)
    assert context.initial_cost > 100 * context.nnet.get_cost(data),\
        'Cost is still too large. Was {}, now {}'.format(context.initial_cost, latest_cost)


@when('I initialize simple neural net with default parameters')
def step(context):
    context.nnet = NeuralNet(layer_sizes=[1, 10, 1])


@when('I train neural network to approximate a function "sin(x)^2"')
def step(context):
    x = []
    y = []

    val_from = -100
    val_to = 100
    incr = 0.1
    farg = val_from
    while farg < val_to:
        v = np.array((1,), float)
        v[0] = math.sin(farg)**2
        x.append(farg)
        y.append(v)
        farg += incr

    examples = (x, y)
    context.nnet.train(examples=examples)


@then('I feed it a set of inputs and get correct results')
def step(context):
    xs = [-10, -5, -1, 0, 1, 5, 10]
    for x in xs:
        y = math.sin(x)**2
        a = context.nnet.feed(np.array([x], float))
        assert abs(a[0] - y) < 0.1, 'a[0]={}, y={}'.format(a[0], y)
