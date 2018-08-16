import numpy as np
import main


def compute_activations_and_zsums(x, neural_net):
    activations = [x]
    zs = []
    a = x
    zs_prime = []
    for layer in neural_net.layers():
        a, z, z_prime = layer.feed_rich(a)
        activations.append(a)
        zs.append(z)
        zs_prime.append(z_prime)

    return activations, zs, zs_prime


class LinkedList:
    class Node:
        def __init__(self, item, next):
            self.next = next
            self.item = item

    class EndOfListError(Exception):
        pass

    def __init__(self, root=None):
        self._root = root

    def prepend(self, item):
        self._root = self.Node(item=item, next=self._root)

    def get_item(self):
        if self.is_empty():
            raise self.EndOfListError('')
        return self._root.item

    def tail(self):
        if self.is_empty():
            raise self.EndOfListError('')
        return LinkedList(root=self._root.next)

    def is_empty(self):
        return self._root is None


class ActivatedLayerList:
    def __init__(self, layers, x):
        self._linked_list = LinkedList()

        activated_layers = []
        a = x
        for layer in layers:
            a_in = a
            a, z, z_prime = layer.feed_rich(a_in)
            activated_layer = main.ActivatedLayer(weights=layer.weights(),
                                                  biases=layer.biases(),
                                                  incoming_activation=a_in,
                                                  activation=a,
                                                  weighted_sum=z,
                                                  weighted_sum_gradient=z_prime)
            activated_layers.append(activated_layer)

        while len(activated_layers) > 0:
            layer = activated_layers.pop()
            self._linked_list.prepend(layer)

    def get_list(self):
        return self._linked_list


def compute_errors(x, y, neural_net):
    def find(layer_list, errors):
        layer = layer_list.get_item()
        z_grad = layer.weighted_sum_gradient
        a = layer.activation

        if layer_list.tail().is_empty():
            nabla = cost_func.get_final_layer_error(a, y, z_grad)
            errors.append(nabla)
            return layer.weights, nabla

        w_next, nabla_next = find(layer_list.tail(), errors)
        nabla = cost_func.get_error_in_layer(nabla_next, w_next, z_grad)
        errors.append(nabla)

        return layer.weights, nabla

    cost_func = neural_net.get_cost_function()

    layer_list = ActivatedLayerList(neural_net.layers(), x=x)
    mylist = layer_list.get_list()

    errors = []
    find(mylist, errors)
    errors.reverse()
    return errors


def back_propagation(x, y, neural_net):
    cost_func = neural_net.get_cost_function()

    weights_gradient = []
    biases_gradient = []
    net_layers = neural_net.number_of_layers() - 1

    activations, weighed_sums, zs_prime = compute_activations_and_zsums(x=x, neural_net=neural_net)

    layer_errors = compute_errors(x, y, neural_net=neural_net)

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
