import numpy as np


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
                weight_grad[layer][i][j] = _weight_gradient(examples, neural_net, layer, i, j)

    for layer in range(nmatrices):
        bias_grad.append(np.zeros(blist[layer].shape))

        rows, = blist[layer].shape
        for row in range(rows):
            bias_grad[layer][row] = _bias_gradient(examples, neural_net, layer, row)

    return weight_grad, bias_grad


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
