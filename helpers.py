import numpy as np
import random
import main


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


def get_final_layer_error(a_last, y, z_last):
    return (a_last - y) * main.sigma_prime(z_last)


def get_weights_gradient(layer_error, previous_layer_activations):
    nabla = layer_error
    a = previous_layer_activations

    return np.outer(nabla, a)


def get_bias_gradient(layer_error):
    return layer_error


def get_error_in_layer(nabla_next, w_next, z):
    return w_next.T.dot(nabla_next) * main.sigma_prime(z)


def compute_activations_and_zsums(x, neural_net):
    net_layers = neural_net.number_of_layers() - 1
    activations = [x]
    zs = []
    a = x
    for layer in range(net_layers):
        a, z = neural_net.feed_into_layer(a, layer=layer)
        activations.append(a)
        zs.append(z)
    return activations, zs


def compute_errors(neural_net, output_activations, expected_output, weighed_sums):
    zs = list(weighed_sums)

    z_L = zs.pop()
    a = output_activations
    y = expected_output
    nabla_L = get_final_layer_error(a, y, z_L)

    net_layers = neural_net.number_of_layers() - 1
    last_layer_index = net_layers - 1

    errors = [nabla_L]

    nabla_next = nabla_L
    wlist = neural_net.weights()

    for layer in range(last_layer_index - 1, -1, -1):
        z = zs.pop()
        w_next = wlist[layer + 1]
        nabla = get_error_in_layer(nabla_next, w_next, z)
        errors.append(nabla)
        nabla_next = nabla

    errors.reverse()
    return errors


def gradients_per_example(x, y, neural_net):
    weights_gradient = []
    biases_gradient = []
    net_layers = neural_net.number_of_layers() - 1

    activations, weighed_sums = compute_activations_and_zsums(x=x, neural_net=neural_net)

    layer_errors = compute_errors(neural_net=neural_net, output_activations=activations[-1],
                                  expected_output=y, weighed_sums=weighed_sums)

    for layer in range(net_layers):
        previous_layer_activations = activations[layer]
        nabla = layer_errors[layer]
        wg = get_weights_gradient(layer_error=nabla,
                                  previous_layer_activations=previous_layer_activations)
        bg = get_bias_gradient(layer_error=nabla)

        weights_gradient.append(wg)
        biases_gradient.append(bg)

    return weights_gradient, biases_gradient


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


def update_total_gradients(summed_gradients_list, new_gradients_list):
    summed_len = len(summed_gradients_list)
    new_len = len(new_gradients_list)
    assert summed_len == new_len

    res_list = []
    for i in range(summed_len):
        res_list.append(summed_gradients_list[i] + new_gradients_list[i])
    return res_list


def average_gradient(gradient_sum, examples_count):
    res_list = []
    for i in range(len(gradient_sum)):
        res_list.append(gradient_sum[i] / float(examples_count))

    return res_list
