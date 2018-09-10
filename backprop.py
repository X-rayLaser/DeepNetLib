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
        cost_func = self._cost_function

        weights_gradient = []
        biases_gradient = []

        pylist = self._propagate_forward()

        layer_errors = self._compute_errors(activated_list=pylist)

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
        activated_layers = []
        a = self._x
        prev_activation_layer = None
        for layer in self._neural_net.layers():
            a_in = a
            a, z, z_prime = layer.feed_rich(a_in)
            activated_layer = ActivatedLayer(weights=layer.weights(),
                                             biases=layer.biases(),
                                             incoming_activation=a_in,
                                             activation=a,
                                             weighted_sum=z,
                                             weighted_sum_gradient=z_prime,
                                             expected_output=self._y,
                                             cost_func=self._cost_function)
            if prev_activation_layer:
                prev_activation_layer.set_next(activated_layer)
            prev_activation_layer = activated_layer
            activated_layers.append(activated_layer)

        return activated_layers

    def _compute_errors(self, activated_list):
        return [layer.get_error() for layer in activated_list]
