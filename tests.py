import unittest
import numpy as np
from main import NeuralNet, quadratic_per_example, quadratic_cost, back_propagation, sigma, sigma_prime
import helpers


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


class NeuralNetTrain(unittest.TestCase):
    def test_gives_correct_output_on_training_data(self):
        nnet = NeuralNet(layer_sizes=[1, 1, 1])

        xes = [np.array([-10], float), np.array([100], float)]
        ys = [np.array([0.5], float), np.array([0.75], float)]

        nnet.train(examples=(xes, ys), epochs=100)

        for i in range(len(xes)):
            res = nnet.feed(xes[i])
            self.assertAlmostEqual(res[0], ys[i][0], places=1)

    def test_gives_correct_output_for_unseen_data(self):
        nnet = NeuralNet(layer_sizes=[1, 1, 1])

        def parabola(x):
            return x**2

        examples = helpers.generate_data(f=parabola, start_value=-1,
                                         end_value=1, step_value=0.05)

        nnet.train(examples=examples, epochs=100)

        xval = -0.5000125
        yval = parabola(xval)

        net_input = np.array([xval], float)
        output = nnet.feed(net_input)
        self.assertAlmostEqual(output[0], yval, places=2)


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
    def compareGrads(self, grad1, grad2):
        nmatrices = len(grad1)
        self.assertEqual(nmatrices, len(grad2))

        for i in range(nmatrices):
            g1 = grad1[i]
            g2 = grad2[i]
            mtx = g1 - g2
            s = np.abs(mtx).sum()
            self.assertLess(s, 0.001)

    def back_propagation_slow(self, examples, neural_net):
        return helpers.back_propagation_slow(examples=examples, neural_net=neural_net)

    def test_back_propagation_slow_per_example(self):
        nnet = NeuralNet(layer_sizes=[1, 1, 1])
        x = np.array([5], float)
        y = np.array([0.25], float)
        examples = ([x], [y])
        w_grad, b_grad = helpers.back_propagation_slow(examples=examples, neural_net=nnet)

        w_grad_expected = [np.array([[0]], float), np.array([[1/32]], float)]
        b_grad_expected = [np.array([[0]], float), np.array([[1/16]], float)]

        self.compareGrads(w_grad, w_grad_expected)
        self.compareGrads(b_grad, b_grad_expected)

    def test_back_propagation(self):
        nnet = NeuralNet(layer_sizes=[4, 15, 10])
        examples = helpers.generate_random_examples(10, 4, 10)

        w_grad1, b_grad1 = back_propagation(examples=examples, neural_net=nnet)
        w_grad2, b_grad2 = self.back_propagation_slow(examples=examples, neural_net=nnet)

        self.compareGrads(grad1=w_grad1, grad2=w_grad2)
        self.compareGrads(grad1=b_grad1, grad2=b_grad2)


if __name__ == '__main__':
    unittest.main()
