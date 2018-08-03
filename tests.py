import unittest
import numpy as np
from main import NeuralNet, GradientDescent, quadratic_per_example, quadratic_cost, sigma, sigma_prime
from backprop import back_propagation
import helpers
import backprop


class NeuralNetInitialization(unittest.TestCase):
    def test_init_with_too_little_layers(self):
        self.assertRaises(NeuralNet.BadArchitecture, lambda: NeuralNet(layer_sizes=[]))
        self.assertRaises(NeuralNet.BadArchitecture, lambda: NeuralNet(layer_sizes=[2]))
        self.assertRaises(NeuralNet.BadArchitecture, lambda: NeuralNet(layer_sizes=[2, 1]))

    def test_init_empty_layers(self):
        self.assertRaises(NeuralNet.BadArchitecture, lambda: NeuralNet(layer_sizes=[0, 1, 1]))
        self.assertRaises(NeuralNet.BadArchitecture, lambda: NeuralNet(layer_sizes=[13, 0, 1]))
        self.assertRaises(NeuralNet.BadArchitecture, lambda: NeuralNet(layer_sizes=[0, 0, 0]))

    def test_valid_init(self):
        NeuralNet(layer_sizes=[1, 1, 1])
        NeuralNet(layer_sizes=[3, 1, 11, 151, 1])

    def test_valid_weights_initialization(self):
        nnet = NeuralNet(layer_sizes=[10, 4, 3])
        w = nnet.weights()
        self.assertEqual(len(w), 2)
        self.assertEqual(w[0].shape, (4, 10))
        self.assertEqual(w[1].shape, (3, 4))

        nnet = NeuralNet(layer_sizes=[100, 25, 15, 10])
        w = nnet.weights()
        self.assertEqual(len(w), 3)
        self.assertEqual(w[0].shape, (25, 100))
        self.assertEqual(w[1].shape, (15, 25))
        self.assertEqual(w[2].shape, (10, 15))

    def test_valid_biases_initialization(self):
        nnet = NeuralNet(layer_sizes=[10, 4, 3])
        b = nnet.biases()
        self.assertEqual(len(b), 2)
        self.assertEqual(b[0].shape, (4,))
        self.assertEqual(b[1].shape, (3,))

        nnet = NeuralNet(layer_sizes=[25, 4, 15, 5])
        b = nnet.biases()
        self.assertEqual(len(b), 3)
        self.assertEqual(b[0].shape, (4,))
        self.assertEqual(b[1].shape, (15,))
        self.assertEqual(b[2].shape, (5,))


class NeuralNetFeedforward(unittest.TestCase):
    def test_feed_after_initialization(self):
        nnet = NeuralNet(layer_sizes=[3, 2, 2])
        x = np.array([1, 9, 323], float)
        a = nnet.feed(x)
        self.assertEqual(a[0], 0.5)
        self.assertEqual(a[1], 0.5)

    def test_feed_into_layer(self):
        nnet = NeuralNet(layer_sizes=[2, 3, 2])

        x = np.array([5, 10], float)
        a, z = nnet.feed_into_layer(x, layer=0)

        self.assertTupleEqual(a.shape, (3,))
        self.assertTupleEqual(z.shape, (3,))
        self.assertEqual(z[0], 0)
        self.assertEqual(z[1], 0)
        self.assertEqual(a[0], 0.5)
        self.assertEqual(a[1], 0.5)

        x = np.array([5, 10, 2], float)
        a, z = nnet.feed_into_layer(x, layer=1)
        self.assertTupleEqual(a.shape, (2,))
        self.assertTupleEqual(z.shape, (2,))
        self.assertEqual(z[0], 0)
        self.assertEqual(z[1], 0)
        self.assertEqual(a[0], 0.5)
        self.assertEqual(a[1], 0.5)


class NeuralNetTrain(unittest.TestCase):
    def test_gives_correct_output_on_training_data(self):
        nnet = NeuralNet(layer_sizes=[1, 1, 1])

        xes = [np.array([-10], float), np.array([100], float)]
        ys = [np.array([0.5], float), np.array([0.75], float)]

        nnet.train(examples=(xes, ys), nepochs=100)

        for i in range(len(xes)):
            res = nnet.feed(xes[i])
            self.assertAlmostEqual(res[0], ys[i][0], places=1)

    def test_gives_correct_output_for_unseen_data(self):
        nnet = NeuralNet(layer_sizes=[1, 10, 1])

        def parabola(x):
            return x**2

        examples = helpers.generate_data(f=parabola, start_value=-0.6,
                                         end_value=-0.4, step_value=0.005)

        nnet.train(examples=examples, nepochs=10)

        xval = -0.5000125
        yval = parabola(xval)

        net_input = np.array([xval], float)
        output = nnet.feed(net_input)
        self.assertAlmostEqual(output[0], yval, places=1)


class NeuralNetCost(unittest.TestCase):
    def test_quadratic_cost(self):
        activations = [np.array([0.7, 0.6], float), np.array([1, 0], float)]
        outputs = [np.array([0.2, 0.5], float), np.array([0, 0], float)]
        c = quadratic_cost(activations=activations, outputs=outputs)
        self.assertAlmostEqual(c, 0.315, places=3)

    def test_quadratic_per_example(self):
        a = np.array([0.5, 0.7], float)
        y = np.array([0.2, 0.1], float)
        c = quadratic_per_example(activation=a, expected_output=y)
        self.assertAlmostEqual(c, 0.225, places=3)

        a = np.array([1, 0], float)
        y = np.array([0, 1], float)
        c = quadratic_per_example(activation=a, expected_output=y)
        self.assertAlmostEqual(c, 1, places=1)

    def test_get_cost_initial(self):
        nnet = NeuralNet(layer_sizes=[1, 1, 1])

        xes = [np.array([-10], float), np.array([100], float)]
        ys = [np.array([0.5], float), np.array([0.75], float)]

        examples = (xes, ys)
        cost = nnet.get_cost(examples)
        self.assertAlmostEqual(cost, 1.0/64, places=4)


class HelpersTests(unittest.TestCase):
    def test_zero_gradients_list(self):
        nnet = NeuralNet(layer_sizes=[3, 5, 4, 1])
        weights_grads, biases_grads = helpers.zero_gradients_list(neural_net=nnet)
        nmatrices = len(weights_grads)
        self.assertEqual(nmatrices, 3)
        self.assertEqual(nmatrices, len(biases_grads))

        self.assertTupleEqual(weights_grads[0].shape, (5, 3))
        self.assertTupleEqual(weights_grads[1].shape, (4, 5))
        self.assertTupleEqual(weights_grads[2].shape, (1, 4))

        self.assertTupleEqual(biases_grads[0].shape, (5,))
        self.assertTupleEqual(biases_grads[1].shape, (4,))
        self.assertTupleEqual(biases_grads[2].shape, (1,))

        for w in weights_grads:
            self.assertAlmostEqual(w.sum(), 0, places=8)

        for b in biases_grads:
            self.assertAlmostEqual(b.sum(), 0, places=8)

    def test_update_total_gradients(self):
        grad_total = [np.array([[100, 120, 130], [50, 10, 60]], int),
                      np.array([[10, 20], [10, 10]], int)]

        grad_last = [np.array([[5, 10, 20], [10, 10, 10]], int),
                     np.array([[1, 1], [2, 4]], int)]

        grad = helpers.update_total_gradients(summed_gradients_list=grad_total,
                                              new_gradients_list=grad_last)

        expected_grad = [np.array([[105, 130, 150], [60, 20, 70]], int),
                         np.array([[11, 21], [12, 14]], int)]

        self.assertEqual(len(grad), 2)
        self.assertTupleEqual(grad[0].shape, (2, 3))
        self.assertTupleEqual(grad[1].shape, (2, 2))

        self.assertTrue(np.all(grad[0] == expected_grad[0]))
        self.assertTrue(np.all(grad[1] == expected_grad[1]))

    def test_average_gradient(self):
        mtx1 = np.array(
            [[5, 10],
             [2, 4]], float
        )

        mtx2 = np.array([[1, 2, 4]], float)
        gradient_sum = [mtx1, mtx2]

        mtx1_expected = np.array(
            [[1, 2],
             [0.4, 0.8]], float
        )

        mtx2_expected = np.array([[0.2, 0.4, 0.8]], float)

        expected_gradient = [mtx1_expected, mtx2_expected]
        grad = helpers.average_gradient(gradient_sum=gradient_sum, examples_count=5)

        self.assertEqual(len(grad), 2)
        self.assertTupleEqual(grad[0].shape, (2, 2))
        self.assertTupleEqual(grad[1].shape, (1, 3))

        self.assertTrue(np.all(grad[0] == expected_gradient[0]))
        self.assertTrue(np.all(grad[1] == expected_gradient[1]))

    def test_get_final_layer_error_for_1_element_vectors(self):
        z_last = np.array([3], float)
        y = np.array([0], float)
        a_last = sigma(z_last)
        nabla = backprop.get_final_layer_error(a_last, y, z_last)
        self.assertAlmostEqual(nabla[0], 0.04, places=2)

        z_last = np.array([-1], float)
        y = np.array([0.5], float)
        a_last = sigma(z_last)
        nabla = backprop.get_final_layer_error(a_last, y, z_last)
        self.assertAlmostEqual(nabla[0], (a_last - y) * sigma_prime(z_last), places=2)

    def test_get_final_layer_error_for_arrays(self):
        z_last = np.array([3, -1], float)
        y = np.array([0, 0.5], float)
        a_last = sigma(z_last)
        nabla = backprop.get_final_layer_error(a_last, y, z_last)

        self.assertAlmostEqual(nabla[0], 0.04, places=2)
        self.assertAlmostEqual(nabla[1], (a_last[1] - y[1]) * sigma_prime(z_last[1]),
                               places=2)

    def test_get_weights_gradient(self):
        layer_error = np.array([3, 5, 10], float)
        activations = np.array([0.5, 0.3], float)
        grad = backprop.get_weights_gradient(layer_error=layer_error,
                                            previous_layer_activations=activations)

        expected_grad = np.array([[1.5, 0.9], [2.5, 1.5], [5., 3.]], float)

        self.assertTupleEqual(grad.shape, expected_grad.shape)
        self.assertTrue(np.allclose(grad, expected_grad))

    def test_get_bias_gradient(self):
        layer_error = np.array([2, 9, 12, 83])
        grad = backprop.get_bias_gradient(layer_error=layer_error)
        self.assertTrue(np.allclose(layer_error, grad))

    def test_get_error_in_layer(self):
        nabla_next = np.array([2, 9, 5], float)
        w_next = np.array([[3, 0], [0, 1], [4, 5]], float)
        z = np.array([2, -1])
        nabla = backprop.get_error_in_layer(nabla_next, w_next, z)

        expected_nabla = np.array([2.72983, 6.6848])
        self.assertTrue(np.allclose(nabla, expected_nabla))

    def test_compute_activations_and_zsums(self):
        nnet = NeuralNet(layer_sizes=[2, 3, 2])
        x = np.array([0.5, 3], float)
        nnet.randomize_parameters()

        a, zs = backprop.compute_activations_and_zsums(x=x, neural_net=nnet)
        expected_activations = nnet.feed(x=x)
        self.assertTrue(np.allclose(a[-1], expected_activations))

    def test_compute_errors(self):
        nnet = NeuralNet(layer_sizes=[1, 2, 1])
        nnet.set_weight(layer=1, row=0, col=0, new_value=-0.5)
        nnet.set_weight(layer=1, row=1, col=0, new_value=1.5)
        nnet.set_weight(layer=2, row=0, col=1, new_value=5)

        nnet.set_bias(layer=1, row=0, new_value=1)
        nnet.set_bias(layer=2, row=0, new_value=-1)

        x = np.array([2], float)

        y = np.array([1], float)

        a, zs = backprop.compute_activations_and_zsums(x=x, neural_net=nnet)

        errors_list = backprop.compute_errors(neural_net=nnet, output_activations=a[-1],
                                             expected_output=y, weighed_sums=zs)

        expected_nabla2 = (a[-1] - y) * sigma_prime(zs[-1])
        expected_nabla1 = backprop.get_error_in_layer(nabla_next=expected_nabla2,
                                                     w_next=np.array([[0, 5]]), z=zs[0])

        self.assertTrue(np.allclose(errors_list[0], expected_nabla1))
        self.assertTrue(np.allclose(errors_list[1], expected_nabla2))


class SetWeight(unittest.TestCase):
    def test_set_weight(self):
        nnet = NeuralNet(layer_sizes=[2, 3, 2])
        nnet.set_weight(layer=1, row=0, col=0, new_value=5)
        nnet.set_weight(layer=1, row=1, col=1, new_value=-2.5)
        nnet.set_weight(layer=2, row=1, col=2, new_value=1.5)

        w = nnet.weights()
        expected_w1 = np.array(
            [[5, 0],
             [0, -2.5],
             [0, 0]], float
        )
        expected_w2 = np.array(
            [[0, 0, 0],
             [0, 0, 1.5]], float
        )

        self.assertTrue(np.allclose(w[0], expected_w1))
        self.assertTrue(np.allclose(w[1], expected_w2))

    def test_raises_exception_for_erroneous_layer(self):
        nnet = NeuralNet(layer_sizes=[2, 1, 2])
        self.assertRaises(NeuralNet.LayerOutOfBound,
                          lambda: nnet.set_weight(layer=0, row=0, col=0, new_value=2)
                          )
        self.assertRaises(NeuralNet.LayerOutOfBound,
                          lambda: nnet.set_weight(layer=-1, row=0, col=0, new_value=2)
                          )
        self.assertRaises(NeuralNet.LayerOutOfBound,
                          lambda: nnet.set_weight(layer=-50, row=0, col=0, new_value=2)
                          )
        self.assertRaises(NeuralNet.LayerOutOfBound,
                          lambda: nnet.set_weight(layer=3, row=0, col=0, new_value=2)
                          )
        self.assertRaises(NeuralNet.LayerOutOfBound,
                          lambda: nnet.set_weight(layer=30, row=0, col=0, new_value=2)
                          )

    def test_raises_exception_for_erroneous_index(self):
        nnet = NeuralNet(layer_sizes=[2, 1, 2])
        self.assertRaises(IndexError,
                          lambda: nnet.set_weight(layer=1, row=1, col=0, new_value=2)
                          )
        self.assertRaises(IndexError,
                          lambda: nnet.set_weight(layer=1, row=0, col=2, new_value=2)
                          )
        self.assertRaises(IndexError,
                          lambda: nnet.set_weight(layer=1, row=1, col=2, new_value=2)
                          )
        self.assertRaises(IndexError,
                          lambda: nnet.set_weight(layer=2, row=2, col=0, new_value=2)
                          )
        self.assertRaises(IndexError,
                          lambda: nnet.set_weight(layer=2, row=1, col=1, new_value=2)
                          )


class SetBias(unittest.TestCase):
    def test_set_bias(self):
        nnet = NeuralNet(layer_sizes=[2, 3, 2])
        nnet.set_bias(layer=1, row=2, new_value=3.3)
        nnet.set_bias(layer=1, row=1, new_value=-1)
        nnet.set_bias(layer=2, row=0, new_value=2)

        expected_bias1 = np.array([0, -1, 3.3], float)
        expected_bias2 = np.array([2, 0], float)

        b = nnet.biases()
        self.assertTrue(np.allclose(b[0], expected_bias1))
        self.assertTrue(np.allclose(b[1], expected_bias2))

    def test_raises_exception_for_erroneous_layer(self):
        nnet = NeuralNet(layer_sizes=[2, 3, 2, 4])
        self.assertRaises(NeuralNet.LayerOutOfBound,
                          lambda: nnet.set_bias(layer=0, row=1, new_value=2)
                          )

        self.assertRaises(NeuralNet.LayerOutOfBound,
                          lambda: nnet.set_bias(layer=4, row=1, new_value=2)
                          )
        self.assertRaises(NeuralNet.LayerOutOfBound,
                          lambda: nnet.set_bias(layer=14, row=1, new_value=2)
                          )

    def test_raises_exception_for_erroneous_index(self):
        nnet = NeuralNet(layer_sizes=[2, 1, 2])
        self.assertRaises(IndexError,
                          lambda: nnet.set_bias(layer=1, row=1, new_value=2)
                          )
        self.assertRaises(IndexError,
                          lambda: nnet.set_bias(layer=2, row=2, new_value=2)
                          )


class SigmoidTests(unittest.TestCase):
    def test_sigma(self):
        self.assertAlmostEqual(sigma(0), 0.5, places=2)
        self.assertAlmostEqual(sigma(50), 1, places=2)
        self.assertAlmostEqual(sigma(-50), 0, places=2)

        self.assertAlmostEqual(sigma(1), 0.731, places=2)
        self.assertAlmostEqual(sigma(-1), 0.2689, places=2)

    def test_sigma_prime(self):
        self.assertAlmostEqual(sigma_prime(0), 0.25, places=3)
        self.assertAlmostEqual(sigma_prime(-50), 0, places=3)
        self.assertAlmostEqual(sigma_prime(50), 0, places=3)

        self.assertAlmostEqual(sigma_prime(50), sigma(50) * (1 - sigma(50)), places=3)


class BackpropagationTests(unittest.TestCase):
    def compare_grads(self, grad1, grad2):
        self.assertTrue(helpers.gradients_equal(grad1, grad2))

    def back_propagation_slow(self, examples, neural_net):
        return helpers.back_propagation_slow(examples=examples, neural_net=neural_net)

    def test_back_propagation_slow(self):
        nnet = NeuralNet(layer_sizes=[1, 1, 1])
        x = np.array([5], float)
        y = np.array([0.25], float)
        examples = ([x], [y])
        w_grad, b_grad = helpers.back_propagation_slow(examples=examples, neural_net=nnet)

        w_grad_expected = [np.array([[0]], float), np.array([[1/32]], float)]
        b_grad_expected = [np.array([[0]], float), np.array([[1/16]], float)]

        self.compare_grads(w_grad, w_grad_expected)
        self.compare_grads(b_grad, b_grad_expected)

    def test_back_propagation_slow_type_array(self):
        nnet = NeuralNet(layer_sizes=[2, 1, 2])
        x = np.array([5, 2], float)
        y = np.array([0.25, 0], float)

        examples = ([x], [y])
        w_grad, b_grad = helpers.back_propagation_slow(examples=examples, neural_net=nnet)
        self.assertIsInstance(w_grad, list)
        self.assertIsInstance(w_grad[0], np.ndarray)
        self.assertIsInstance(w_grad[1], np.ndarray)

        self.assertIsInstance(b_grad, list)
        self.assertIsInstance(b_grad[0], np.ndarray)
        self.assertIsInstance(b_grad[1], np.ndarray)

    def test_back_propagation_slow_shape(self):
        nnet = NeuralNet(layer_sizes=[3, 2, 2, 5])
        x = np.array([5, 2, -0.5], float)
        y = np.array([0.25, 0, 0, 0.7, 0.2], float)
        examples = ([x], [y])
        w_grad, b_grad = helpers.back_propagation_slow(examples=examples, neural_net=nnet)
        self.assertEqual(len(w_grad), 3)
        self.assertEqual(len(b_grad), 3)
        self.assertTupleEqual(w_grad[0].shape, (2, 3))
        self.assertTupleEqual(w_grad[1].shape, (2, 2))
        self.assertTupleEqual(w_grad[2].shape, (5, 2))

        self.assertTupleEqual(b_grad[0].shape, (2,))
        self.assertTupleEqual(b_grad[1].shape, (2,))
        self.assertTupleEqual(b_grad[2].shape, (5,))

    def test_back_propagation(self):
        nnet = NeuralNet(layer_sizes=[4, 15, 10])
        examples = helpers.generate_random_examples(10, 4, 10)

        w_grad1, b_grad1 = back_propagation(examples=examples, neural_net=nnet)
        w_grad2, b_grad2 = self.back_propagation_slow(examples=examples, neural_net=nnet)

        self.compare_grads(grad1=w_grad1, grad2=w_grad2)
        self.compare_grads(grad1=b_grad1, grad2=b_grad2)

    def test_back_propagation_type_array(self):
        nnet = NeuralNet(layer_sizes=[2, 1, 2])
        x = np.array([5, 2], float)
        y = np.array([0.25, 0], float)

        examples = ([x], [y])
        w_grad, b_grad = back_propagation(examples=examples, neural_net=nnet)
        self.assertIsInstance(w_grad, list)
        self.assertIsInstance(w_grad[0], np.ndarray)
        self.assertIsInstance(w_grad[1], np.ndarray)

        self.assertIsInstance(b_grad, list)
        self.assertIsInstance(b_grad[0], np.ndarray)
        self.assertIsInstance(b_grad[1], np.ndarray)

    def test_back_propagation_returns_correct_gradient_shape(self):
        nnet = NeuralNet(layer_sizes=[3, 2, 2, 5])
        x = np.array([5, 2, -0.5], float)
        y = np.array([0.25, 0, 0, 0.7, 0.2], float)
        examples = ([x], [y])
        w_grad, b_grad = helpers.back_propagation_slow(examples=examples, neural_net=nnet)
        self.assertEqual(len(w_grad), 3)
        self.assertEqual(len(b_grad), 3)
        self.assertTupleEqual(w_grad[0].shape, (2, 3))
        self.assertTupleEqual(w_grad[1].shape, (2, 2))
        self.assertTupleEqual(w_grad[2].shape, (5, 2))

        self.assertTupleEqual(b_grad[0].shape, (2,))
        self.assertTupleEqual(b_grad[1].shape, (2,))
        self.assertTupleEqual(b_grad[2].shape, (5,))


class GradientDescentTest(unittest.TestCase):
    def setUp(self):
        x = np.array([5, 2], float)
        y = np.array([0.25, 0, 1], float)
        self.examples = ([x], [y])
        nnet = NeuralNet(layer_sizes=[2, 3, 3])
        nnet.randomize_parameters()
        self.nnet = nnet
        self.grad_descent = GradientDescent(neural_net=nnet)

    def test_init(self):
        nnet = NeuralNet(layer_sizes=[2, 3, 5])
        grad_descent = GradientDescent(neural_net=nnet)

    def test_update_weights_decreases_cost(self):
        cost_before = self.nnet.get_cost(self.examples)

        for i in range(5):
            w_grad, b_grad = back_propagation(examples=self.examples, neural_net=self.nnet)
            self.grad_descent.update_weights(weight_gradient=w_grad)
            self.grad_descent.update_biases(bias_gradient=b_grad)

        cost_after = self.nnet.get_cost(self.examples)
        self.assertLess(cost_after, cost_before)

    def test_update_biases_decreases_cost(self):
        cost_before = self.nnet.get_cost(self.examples)

        for i in range(5):
            w_grad, b_grad = back_propagation(examples=self.examples, neural_net=self.nnet)
            self.grad_descent.update_biases(bias_gradient=b_grad)

        cost_after = self.nnet.get_cost(self.examples)
        self.assertLess(cost_after, cost_before)

    def test_update_with_multiple_examples(self):
        self.nnet.randomize_parameters()

        cost_before = self.nnet.get_cost(self.examples)

        self.examples = ([np.array([5, 2], float), np.array([5, 22], float)],
                         [np.array([0.25, 0, 1], float), np.array([0.5, 1, 0], float)])

        for i in range(15):
            w_grad, b_grad = back_propagation(examples=self.examples, neural_net=self.nnet)
            self.grad_descent.update_weights(weight_gradient=w_grad)
            self.grad_descent.update_biases(bias_gradient=b_grad)

        cost_after = self.nnet.get_cost(self.examples)
        self.assertLess(cost_after, cost_before)

    def test_training_epoch_1_example(self):
        cost_before = self.nnet.get_cost(self.examples)
        self.grad_descent.training_epoch(examples=self.examples)
        cost_after = self.nnet.get_cost(self.examples)
        self.assertLess(cost_after, cost_before)

    def test_training_epoch_2_examples(self):
        self.examples = ([np.array([5, 2], float), np.array([5, 22], float)],
                         [np.array([0.25, 0, 1], float), np.array([0.5, 1, 0], float)])
        cost_before = self.nnet.get_cost(self.examples)
        self.grad_descent.training_epoch(examples=self.examples)
        cost_after = self.nnet.get_cost(self.examples)
        self.assertLess(cost_after, cost_before)

    def test_training_with_initially_random_parameters(self):
        self.nnet.randomize_parameters()
        cost_before = self.nnet.get_cost(self.examples)
        self.grad_descent.training_epoch(examples=self.examples)
        cost_after = self.nnet.get_cost(self.examples)
        self.assertLess(cost_after, cost_before)

    def test_train_for_few_epoch(self):
        cost_before = self.nnet.get_cost(self.examples)
        self.grad_descent.train(examples=self.examples, nepochs=2)
        cost_after = self.nnet.get_cost(self.examples)
        self.assertLess(cost_after, cost_before)


if __name__ == '__main__':
    unittest.main()
