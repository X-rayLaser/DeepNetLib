import numpy as np


def compute_activations_and_zsums(x, neural_net):
    activations = [x]
    zs = []
    a = x
    for layer in neural_net.layers():
        a, z = layer.feed(a)
        activations.append(a)
        zs.append(z)

    return activations, zs


def compute_errors(neural_net, output_activations, expected_output, weighed_sums):
    cost_func = neural_net.get_cost_function()
    activation_function = neural_net.get_activation_function()

    zs = list(weighed_sums)

    z_L = zs.pop()
    a = output_activations
    y = expected_output
    nabla_L = cost_func.get_final_layer_error(a, y, z_L, activation_function=activation_function)

    net_layers = neural_net.number_of_layers() - 1
    last_layer_index = net_layers - 1

    errors = [nabla_L]

    nabla_next = nabla_L
    wlist = neural_net.weights()

    for layer in range(last_layer_index - 1, -1, -1):
        z = zs.pop()
        w_next = wlist[layer + 1]
        nabla = cost_func.get_error_in_layer(nabla_next, w_next, z,
                                             activation_function=activation_function)
        errors.append(nabla)
        nabla_next = nabla

    errors.reverse()
    return errors


def back_propagation(x, y, neural_net):
    cost_func = neural_net.get_cost_function()

    weights_gradient = []
    biases_gradient = []
    net_layers = neural_net.number_of_layers() - 1

    activations, weighed_sums = compute_activations_and_zsums(x=x, neural_net=neural_net)

    layer_errors = compute_errors(neural_net=neural_net, output_activations=activations[-1],
                                  expected_output=y, weighed_sums=weighed_sums)

    for layer in range(net_layers):
        previous_layer_activations = activations[layer]
        nabla = layer_errors[layer]
        wg = cost_func.get_weights_gradient(layer_error=nabla,
                                  previous_layer_activations=previous_layer_activations)
        bg = cost_func.get_bias_gradient(layer_error=nabla)

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
