import numpy as np
import helpers
import main


def back_propagation(examples, neural_net):
    xes, ys = examples
    examples_count = len(ys)

    weights_grad, biases_grad = zero_gradients_list(neural_net)

    for i in range(examples_count):
        x = xes[i]
        y = ys[i]
        wgrad, bgrad = gradients_per_example(x, y, neural_net=neural_net)
        weights_grad = update_total_gradients(summed_gradients_list=weights_grad,
                                                      new_gradients_list=wgrad)
        biases_grad = update_total_gradients(summed_gradients_list=biases_grad,
                                                     new_gradients_list=bgrad)

    weights_grad = average_gradient(weights_grad, examples_count)
    biases_grad = average_gradient(biases_grad, examples_count)

    return weights_grad, biases_grad


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
