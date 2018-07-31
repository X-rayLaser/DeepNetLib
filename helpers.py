import numpy as np


def generate_data(f, start_value, end_value, step_value):
    x = []
    y = []

    farg = start_value
    while farg < end_value:
        v = np.array((1,), float)
        v[0] = f(farg)
        x.append(np.array([farg], float))
        y.append(v)
        farg += step_value
    return x, y


def back_propagation_slow(examples, neural_net):
    wlist = neural_net.weights()
    blist = neural_net.biases()

    nmatrices = len(wlist)

    weight_grad = []
    bias_grad = []

    epsilon = 10 ** (-5)
    for i in range(nmatrices):
        weights = wlist[i]

    return [], []


def back_propagation_slow_per_example(x, y, neural_net):
    return [], []