import numpy as np
import main


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

    def to_pylist(self):
        pylist = []
        linked_list = self
        while not linked_list.is_empty():
            pylist.append(linked_list.get_item())
            linked_list = linked_list.tail()

        return pylist


class BackPropagation:
    def __init__(self, x, y, neural_net):
        self._x = x
        self._y = y
        self._neural_net = neural_net

    def back_propagate(self):
        neural_net = self._neural_net
        x = self._x
        y = self._y
        cost_func = neural_net.get_cost_function()

        weights_gradient = []
        biases_gradient = []

        mylist = self._propagate_forward()
        pylist = mylist.to_pylist()

        layer_errors = self._compute_errors(activated_list=mylist)

        for activated_layer, nabla in zip(pylist, layer_errors):
            a_in = activated_layer.incoming_activation
            wg = cost_func.get_weights_gradient(
                layer_error=nabla,
                previous_layer_activations=a_in
            )
            bg = cost_func.get_bias_gradient(layer_error=nabla)

            weights_gradient.append(wg)
            biases_gradient.append(bg)

        return weights_gradient, biases_gradient

    def _propagate_forward(self):
        layers = self._neural_net.layers()
        x = self._x

        linked_list = LinkedList()

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
            linked_list.prepend(layer)
        return linked_list

    def _compute_errors(self, activated_list):
        x = self._x
        y = self._y
        neural_net = self._neural_net

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

        errors = []
        find(activated_list, errors)
        errors.reverse()
        return errors


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
