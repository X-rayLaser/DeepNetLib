"""
:class BackPropagation: implements back propagation algorithm
"""
from structures import LinkedList, ActivatedLayer


class BackPropagation:
    """
    Implement back propagation algorithm on a single training example (x, y)
    
    :method back_propagate
    """
    def __init__(self, x, y, neural_net, cost_function):
        """
        Create an instance of a class
        
        :param x: input to the neural net, numpy 1d array
        :param y: expected output of the neural net, numpy 1d array
        :param neural_net: instance of NeuralNet type
        """
        self._x = x
        self._y = y
        self._neural_net = neural_net
        self._cost_function = cost_function

    def back_propagate(self):
        """
        Run a back propagation algorithm using for a given net
        :return: a tuple of 2d numpy arrays gradients for weights and biases
        """
        neural_net = self._neural_net
        cost_func = self._cost_function

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
            activated_layer = ActivatedLayer(weights=layer.weights(),
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
        errors = LinkedList()
        self._compute(activated_list, errors)
        return errors.to_pylist()

    def _compute(self, layer_list, errors):
        y = self._y
        neural_net = self._neural_net
        cost_func = self._cost_function

        layer = layer_list.get_item()
        z_grad = layer.weighted_sum_gradient
        a = layer.activation

        if layer_list.tail().is_empty():
            nabla = cost_func.get_final_layer_error(a, y, z_grad)
            errors.prepend(nabla)
            return layer.weights, nabla

        w_next, nabla_next = self._compute(layer_list.tail(), errors)
        nabla = cost_func.get_error_in_layer(nabla_next, w_next, z_grad)
        errors.prepend(nabla)

        return layer.weights, nabla
