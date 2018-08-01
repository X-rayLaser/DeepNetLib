import numpy as np
import random


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


def random_input_vector(size):
    sr = random.SystemRandom()
    v = np.zeros((size,), float)
    for i in range(size):
        v[i] = sr.random() * 100 - 50
    return v


def random_output_vector(size):
    sr = random.SystemRandom()
    v = np.zeros((size,), float)
    for i in range(size):
        v[i] = sr.random()
    return v


def generate_random_examples(examples_number, input_size, output_size):
    sr = random.SystemRandom()
    xes = []
    ys = []
    for i in range(examples_number):
        x = random_input_vector(input_size)
        y = random_output_vector(output_size)
        xes.append(x)
        ys.append(y)

    return xes, ys


def _weight_gradient(examples, neural_net, layer, i, j):
    epsilon = 10 ** (-5)
    wlist = neural_net.weights()
    weights = wlist[layer]

    neural_net.set_weight(layer=layer+1, row=i, col=j,
                          new_value=weights[i, j] - epsilon)
    cost_minus = neural_net.get_cost(examples)
    neural_net.set_weight(layer=layer + 1, row=i, col=j,
                          new_value=weights[i, j] + 2 * epsilon)
    cost_plus = neural_net.get_cost(examples)

    derivative = (cost_plus - cost_minus) / (2 * epsilon)

    neural_net.set_weight(layer=layer + 1, row=i, col=j,
                          new_value=weights[i, j] - epsilon)

    return derivative


def _bias_gradient(examples, neural_net, layer, row):
    epsilon = 10 ** (-5)
    blist = neural_net.biases()

    biases = blist[layer]

    neural_net.set_bias(layer=layer+1, row=row, new_value=biases[row] - epsilon)
    cost_minus = neural_net.get_cost(examples)
    neural_net.set_bias(layer=layer+1, row=row, new_value=biases[row] + 2 * epsilon)
    cost_plus = neural_net.get_cost(examples)
    derivative = (cost_plus - cost_minus) / (2 * epsilon)
    neural_net.set_bias(layer=layer+1, row=row, new_value=biases[row] - epsilon)
    return derivative


def back_propagation_slow(examples, neural_net):
    """approximated numerical partial derivatives"""
    wlist = neural_net.weights()
    blist = neural_net.biases()

    nmatrices = len(wlist)
    weight_grad = []
    bias_grad = []

    for layer in range(nmatrices):
        weight_grad.append(np.zeros(wlist[layer].shape))

        rows, cols = wlist[layer].shape
        for i in range(rows):
            for j in range(cols):
                weight_grad[layer] = _weight_gradient(examples, neural_net, layer, i, j)

    for layer in range(nmatrices):
        bias_grad.append(np.zeros(blist[layer].shape))

        rows, = blist[layer].shape
        for row in range(rows):
            bias_grad[layer] = _bias_gradient(examples, neural_net, layer, row)

    return weight_grad, bias_grad


def gradients_equal(grad1, grad2):
    nmatrices = len(grad1)

    if nmatrices != len(grad2):
        return False

    for i in range(nmatrices):
        g1 = grad1[i]
        g2 = grad2[i]
        mtx = g1 - g2
        s = np.abs(mtx).sum()
        if s > 0.001:
            return False
    return True


def gradients_per_example(x, y, neural_net):
    pass


def zero_gradients_list(neural_net):
    weights_grad = []
    biases_grad = []
    wlist = neural_net.weights()
    blist = neural_net.biases()

    for i in range(len(wlist)):
        wshape = wlist[i].shape
        weights_grad.append(np.zeros(wshape))

        bshape = blist[i].shape
        biases_grad.append(np.zeros(bshape))

    return weights_grad, biases_grad


def update_total_gradients(summed_gradients, new_gradients):
    pass


def average_gradient():
    pass
