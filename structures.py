import numpy as np


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


class ActivatedLayer:
    """
    Linked collection of objects each of which encapsulates activations in some layer.
    
    It is used by back propagation routine to easily calculate gradients of the
    cost function with respect to weights and biases.
    
    Each instance of this class is related to the corresponding Layer instance. 
    
    Public methods:
        set_next: set next layer in the chain
        get_weights_gradient: compute a matrix of derivatives w.r.t weights
        get_bias_gradient: compute a vector of derivatives w.r.t. biases
    """
    def __init__(self, weights, biases, incoming_activation,
                 activation, weighted_sum, weighted_sum_gradient, expected_output, cost_func):
        self.weights = weights
        self.biases = biases
        self.incoming_activation = incoming_activation
        self.activation = activation
        self.weighted_sum = weighted_sum
        self.weighted_sum_gradient = weighted_sum_gradient

        self._next = None
        self._expected_output = expected_output
        self._cost_func = cost_func
        self._error = None

    def set_next(self, next_layer):
        self._next = next_layer

    def get_error(self):
        if self._error:
            return self._error

        if self._next:
            next_layer = self._next
            nabla_next = next_layer.get_error()
            error = next_layer.weights.T.dot(nabla_next) * self.weighted_sum_gradient
        else:
            error = self._cost_func.get_final_layer_error(
                self.activation, self._expected_output, self.weighted_sum_gradient
            )
        return error

    def get_weights_gradient(self):
        """
        Calculate all partial derivatives of loss function with respect to weights in this layer.
        """
        nabla = self.get_error()
        a = self.incoming_activation

        return np.outer(nabla, a) + self._cost_func.get_lambda() * self.weights

    def get_bias_gradient(self):
        """
        Calculate a vector of partial derivatives of loss function with respect to biases in a layer.
        """
        return self.get_error()
