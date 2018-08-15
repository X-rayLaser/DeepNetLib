import unittest
import numpy as np
from cost_functions import QuadraticCost, CrossEntropyCost, RegularizedCost


class RegularizedGradientsTests(unittest.TestCase):
    def test_get_weights_gradient_for_single_example(self):
        activations = [np.array([-1, 1], float)]
        layer_error = [np.array([0, 1], float)]

        reglambda = 1
        weights = np.array([[3, -1], [0.5, 2]], float)
        regcost = RegularizedCost(cost_function=QuadraticCost(),
                                  regularization_parameter=reglambda,
                                  weights=weights)

        greg = regcost.get_weights_gradient(layer_error=layer_error,
                                            previous_layer_activations=activations,
                                            weights=weights)

        quadcost = QuadraticCost()
        gquad = quadcost.get_weights_gradient(layer_error=layer_error,
                                              previous_layer_activations=activations,
                                              weights=weights)

        for i in range(greg.shape[0]):
            for j in range(greg.shape[1]):
                expected_value = gquad[i, j] + reglambda * weights[i, j]
                self.assertAlmostEqual(greg[i, j], expected_value)
