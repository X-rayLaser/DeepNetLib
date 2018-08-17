import numpy as np
from backprop import BackPropagation


class GradientCalculator:
    pass


class NumericalCalculator:
    def __init__(self, examples, neural_net):
        self._examples = examples
        self._neural_net = neural_net

    def compute_gradients(self):
        """approximated numerical partial derivatives"""
        wlist = self._neural_net.weights()
        blist = self._neural_net.biases()

        nmatrices = len(wlist)
        weight_grad = []
        bias_grad = []

        for layer in range(nmatrices):
            weight_grad.append(np.zeros(wlist[layer].shape))

            rows, cols = wlist[layer].shape
            for i in range(rows):
                for j in range(cols):
                    weight_grad[layer][i][j] = self._weight_gradient(layer, i, j)

        for layer in range(nmatrices):
            bias_grad.append(np.zeros(blist[layer].shape))

            rows, = blist[layer].shape
            for row in range(rows):
                bias_grad[layer][row] = self._bias_gradient(layer, row)

        return weight_grad, bias_grad

    def _weight_gradient(self, layer, i, j):
        neural_net = self._neural_net
        examples = self._examples

        epsilon = 10 ** (-5)
        wlist = neural_net.weights()
        weights = wlist[layer]

        neural_net.set_weight(layer=layer + 1, row=i, col=j,
                              new_value=weights[i, j] - epsilon)
        cost_minus = neural_net.get_cost(examples)
        neural_net.set_weight(layer=layer + 1, row=i, col=j,
                              new_value=weights[i, j] + 2 * epsilon)
        cost_plus = neural_net.get_cost(examples)

        derivative = (cost_plus - cost_minus) / (2 * epsilon)

        neural_net.set_weight(layer=layer + 1, row=i, col=j,
                              new_value=weights[i, j] - epsilon)

        return derivative

    def _bias_gradient(self, layer, row):
        neural_net = self._neural_net
        examples = self._examples
        epsilon = 10 ** (-5)
        blist = neural_net.biases()

        biases = blist[layer]

        neural_net.set_bias(layer=layer + 1, row=row, new_value=biases[row] - epsilon)
        cost_minus = neural_net.get_cost(examples)
        neural_net.set_bias(layer=layer + 1, row=row, new_value=biases[row] + 2 * epsilon)
        cost_plus = neural_net.get_cost(examples)
        derivative = (cost_plus - cost_minus) / (2 * epsilon)
        neural_net.set_bias(layer=layer + 1, row=row, new_value=biases[row] - epsilon)
        return derivative


class BackPropagationBasedCalculator:
    def __init__(self, examples, neural_net):
        self._examples = examples
        self._neural_net = neural_net

    def compute_gradients(self):
        xes, ys = self._examples
        examples_count = len(ys)
        reglambda = self._neural_net.get_cost_function().get_lambda()

        weights_grad, biases_grad = self._zero_gradients_list(self._neural_net)

        for i in range(examples_count):
            x = xes[i]
            y = ys[i]
            backprop = BackPropagation(x, y, neural_net=self._neural_net)
            wgrad, bgrad = backprop.back_propagate()
            weights_grad = self._update_total_gradients(summed_gradients_list=weights_grad,
                                                        new_gradients_list=wgrad)
            biases_grad = self._update_total_gradients(summed_gradients_list=biases_grad,
                                                       new_gradients_list=bgrad)

        weights_grad = self._average_gradient(weights_grad, examples_count)
        biases_grad = self._average_gradient(biases_grad, examples_count)

        weights = self._neural_net.weights()
        for i in range(len(weights_grad)):
            weights_grad[i] = weights_grad[i] + reglambda / float(examples_count) * weights[i]

        return weights_grad, biases_grad

    def _zero_gradients_list(self, neural_net):
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

    def _update_total_gradients(self, summed_gradients_list, new_gradients_list):
        summed_len = len(summed_gradients_list)
        new_len = len(new_gradients_list)
        assert summed_len == new_len

        res_list = []
        for i in range(summed_len):
            res_list.append(summed_gradients_list[i] + new_gradients_list[i])
        return res_list

    def _average_gradient(self, gradient_sum, examples_count):
        res_list = []
        for i in range(len(gradient_sum)):
            res_list.append(gradient_sum[i] / float(examples_count))

        return res_list
