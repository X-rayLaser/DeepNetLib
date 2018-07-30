from main import NeuralNet
import numpy as np
import math


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
