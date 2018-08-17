import numpy as np
import math


class CostFunction:
    def get_error_in_layer(self, nabla_next, w_next, z_gradient):
        return w_next.T.dot(nabla_next) * z_gradient

    def get_weights_gradient(self, layer_error, previous_layer_activations):
        nabla = layer_error
        a = previous_layer_activations

        return np.outer(nabla, a)

    def get_final_layer_error(self, activation_last, expected_output, z_gradient_last):
        a_last = activation_last
        y = expected_output
        return (a_last - y) * z_gradient_last

    def get_bias_gradient(self, layer_error):
        return layer_error

    def get_lambda(self):
        return 0


class QuadraticCost(CostFunction):
    def _quadratic_per_example(self, activation, expected_output):
        v = activation - expected_output
        return v.dot(v) / 2.0

    def _quadratic_cost(self, activations, outputs):
        vector_len = len(activations)

        s = 0
        for i in range(vector_len):
            s += self._quadratic_per_example(activation=activations[i],
                                            expected_output=outputs[i])
        return s / vector_len

    def compute_cost(self, activations, outputs):
        return self._quadratic_cost(activations=activations, outputs=outputs)


class CrossEntropyCost(CostFunction):
    def compute_cost(self, activations, outputs):
        ce = 0
        num_of_examples = len(outputs)
        for i in range(num_of_examples):
            ce += self._per_example(activations[i], outputs[i])
        return ce / num_of_examples

    def get_final_layer_error(self, activation_last, expected_output, z_gradient):
        a_last = activation_last
        y = expected_output
        return a_last - y

    def _between_numbers(self, a, y):
        """Cross entropy between 2 numbers, a and y defined on [0, 1]"""
        if a == 0 and y == 0:
            return 0

        if a == 1 and y == 1:
            return 0

        if a == 0 or a == 1:
            return np.finfo(float).max

        return -y * math.log(a) - (1 - y) * math.log(1 - a, math.e)

    def _per_example(self, a, y):
        ncomponents = a.shape[0]
        ce = 0
        for i in range(ncomponents):
            ce += self._between_numbers(a[i], y[i])

        return ce


class RegularizedCost(CostFunction):
    def __init__(self, cost_function, regularization_parameter, weights):
        self._cost_function = cost_function
        self._reglambda = regularization_parameter
        self._weights = weights

    def compute_cost(self, activations, outputs):
        n = len(outputs)

        square_sum = sum([(w ** 2).sum() for w in self._weights])

        reg_term = self._reglambda / float(2 * n) * square_sum
        old_cost = self._cost_function.compute_cost(activations=activations,
                                                    outputs=outputs)
        return old_cost + reg_term

    def get_lambda(self):
        return self._reglambda
