import unittest
import json
import os
import numpy as np
from main import NeuralNet
from activation_functions import Sigmoid, Rectifier, Softmax


class SetLayerWeights(unittest.TestCase):
    def test_with_single_hidden_layer(self):
        nnet = NeuralNet(layer_sizes=[1, 2, 2])
        weights = [[3], [-3.5]]
        nnet.set_layer_weights(layer=1, weights=np.array(weights, float))
        self.assertEqual(nnet.weights()[0].tolist(), weights)
        self.assertEqual(nnet.weights()[1].tolist(), [[0, 0], [0, 0]])

        weights2 = [[0, 10], [1, 1]]
        nnet.set_layer_weights(layer=2, weights=np.array(weights2, float))
        self.assertEqual(nnet.weights()[0].tolist(), weights)
        self.assertEqual(nnet.weights()[1].tolist(), weights2)

    def test_with_2_hidden_layers(self):
        nnet = NeuralNet(layer_sizes=[1, 2, 2, 3])
        weights2 = [[0, 10], [1, 1]]
        weights3 = [[0, 10], [1, 1], [-1.5, 2]]

        nnet.set_layer_weights(layer=2, weights=np.array(weights2, float))
        nnet.set_layer_weights(layer=3, weights=np.array(weights3, float))

        self.assertEqual(nnet.weights()[0].tolist(), [[0], [0]])
        self.assertEqual(nnet.weights()[1].tolist(), weights2)
        self.assertEqual(nnet.weights()[2].tolist(), weights3)

    def test_raises_exception_for_array_with_wrong_layer_index(self):
        nnet = NeuralNet(layer_sizes=[1, 2, 2])
        self.assertRaises(NeuralNet.LayerOutOfBound,
                          lambda: nnet.set_layer_weights(layer=-1, weights=[]))
        self.assertRaises(NeuralNet.LayerOutOfBound,
                          lambda: nnet.set_layer_weights(layer=0, weights=[]))
        self.assertRaises(NeuralNet.LayerOutOfBound,
                          lambda: nnet.set_layer_weights(layer=3, weights=[]))
        self.assertRaises(NeuralNet.LayerOutOfBound,
                          lambda: nnet.set_layer_weights(layer=30, weights=[]))

    def test_raises_exception_for_array_with_wrong_dimension(self):
        nnet = NeuralNet(layer_sizes=[1, 2, 2])

        self.assertRaises(
            NeuralNet.InvalidMatrixDimensions,
            lambda: nnet.set_layer_weights(layer=1, weights=np.zeros((2, 2)))
        )

        self.assertRaises(
            NeuralNet.InvalidMatrixDimensions,
            lambda: nnet.set_layer_weights(layer=2, weights=np.zeros((4, 3)))
        )


class SetLayerBiases(unittest.TestCase):
    def test_with_single_hidden_layer(self):
        nnet = NeuralNet(layer_sizes=[1, 2, 3])
        biases1 = [9, -2.5]
        biases2 = [1.5, 0, 5]

        nnet.set_layer_biases(layer=1, bias_vector=np.array(biases1, float))
        nnet.set_layer_biases(layer=2, bias_vector=np.array(biases2, float))

        self.assertEqual(nnet.biases()[0].tolist(), biases1)
        self.assertEqual(nnet.biases()[1].tolist(), biases2)

    def test_with_2_hidden_layers(self):
        nnet = NeuralNet(layer_sizes=[1, 2, 2, 1])
        biases2 = [9, -2.5]
        biases3 = [1.5]

        nnet.set_layer_biases(layer=2, bias_vector=np.array(biases2, float))
        nnet.set_layer_biases(layer=3, bias_vector=np.array(biases3, float))

        self.assertEqual(nnet.biases()[0].tolist(), [0, 0])
        self.assertEqual(nnet.biases()[1].tolist(), biases2)
        self.assertEqual(nnet.biases()[2].tolist(), biases3)

    def test_raises_exception_for_array_with_wrong_layer_index(self):
        nnet = NeuralNet(layer_sizes=[1, 2, 2])
        self.assertRaises(NeuralNet.LayerOutOfBound,
                          lambda: nnet.set_layer_biases(layer=-1, bias_vector=[]))
        self.assertRaises(NeuralNet.LayerOutOfBound,
                          lambda: nnet.set_layer_biases(layer=0, bias_vector=[]))
        self.assertRaises(NeuralNet.LayerOutOfBound,
                          lambda: nnet.set_layer_biases(layer=3, bias_vector=[]))
        self.assertRaises(NeuralNet.LayerOutOfBound,
                          lambda: nnet.set_layer_biases(layer=30, bias_vector=[]))

    def test_raises_exception_for_array_with_wrong_dimension(self):
        nnet = NeuralNet(layer_sizes=[1, 2, 2])

        self.assertRaises(
            NeuralNet.InvalidMatrixDimensions,
            lambda: nnet.set_layer_biases(layer=1, bias_vector=np.zeros((3,)))
        )

        self.assertRaises(
            NeuralNet.InvalidMatrixDimensions,
            lambda: nnet.set_layer_biases(layer=2, bias_vector=np.zeros((4,)))
        )


class SetActivationFunctionTests(unittest.TestCase):
    def test_feed_outputs_vector_of_zeros(self):
        nnet = NeuralNet(layer_sizes=[2, 4, 3])
        nnet.set_activation_function(activation=Rectifier)
        a = nnet.feed(np.array([2, 10], float))
        self.assertEqual(a.tolist(), [0, 0, 0])

    def test_feed_outputs_correct_vector(self):
        nnet = NeuralNet(layer_sizes=[2, 4, 4])
        nnet.set_layer_biases(layer=2, bias_vector=np.array([-2, -1, 1, 10], float))

        nnet.set_activation_function(activation=Rectifier)
        a = nnet.feed(np.array([2, 10], float))
        self.assertEqual(a.tolist(), [0, 0, 1, 10])


class SetOutputActivationFunctionTests(unittest.TestCase):
    def test_for_rectifier_feed_outputs_vector_of_zeros(self):
        nnet = NeuralNet(layer_sizes=[2, 4, 3])
        nnet.set_activation_function(activation=Sigmoid)
        nnet.set_output_activation_function(activation=Rectifier)
        a = nnet.feed(np.array([2, 10], float))
        self.assertEqual(a.tolist(), [0, 0, 0])

    def test_for_rectifier(self):
        nnet = NeuralNet(layer_sizes=[2, 4, 4])
        nnet.set_activation_function(activation=Sigmoid)
        nnet.set_layer_biases(layer=2, bias_vector=np.array([-2, -1, 1, 10], float))

        nnet.set_output_activation_function(activation=Rectifier)
        a = nnet.feed(np.array([2, 10], float))
        self.assertEqual(a.tolist(), [0, 0, 1, 10])

    def test_for_softmax(self):
        nnet = NeuralNet(layer_sizes=[2, 4, 2])
        nnet.set_output_activation_function(activation=Rectifier)

        nnet.set_layer_biases(layer=2, bias_vector=np.array([3, 3], float))
        nnet.set_output_activation_function(activation=Softmax)
        a = nnet.feed(np.array([2, 10], float))
        self.assertEqual(a.tolist(), [0.5, 0.5])


class GetActivationFunctionTests(unittest.TestCase):
    def test_for_default_activation(self):
        nnet = NeuralNet(layer_sizes=[2, 4, 3])
        activation_object = nnet.get_activation_function()

        self.assertEqual(activation_object.__class__, Sigmoid.__class__)

    def test_for_sigmoid(self):
        nnet = NeuralNet(layer_sizes=[2, 4, 3])
        nnet.set_activation_function(Sigmoid)
        activation_object = nnet.get_activation_function()

        self.assertEqual(activation_object, Sigmoid)

    def test_for_rectifier(self):
        nnet = NeuralNet(layer_sizes=[2, 4, 3])
        nnet.set_activation_function(Rectifier)
        activation_object = nnet.get_activation_function()

        self.assertEqual(activation_object, Rectifier)


class GetOutputLayerActivationFunctionTests(unittest.TestCase):
    def test_for_sigmoid(self):
        nnet = NeuralNet(layer_sizes=[2, 4, 3])
        from activation_functions import Sigmoid
        nnet.set_output_activation_function(Sigmoid)
        activation_object = nnet.get_output_activation_function()

        self.assertEqual(activation_object, Sigmoid)

    def test_for_rectifier(self):
        nnet = NeuralNet(layer_sizes=[2, 4, 3])
        nnet.set_output_activation_function(Rectifier)
        activation_object = nnet.get_output_activation_function()

        self.assertEqual(activation_object, Rectifier)

    def test_for_softmax(self):
        nnet = NeuralNet(layer_sizes=[2, 4, 3])
        from activation_functions import Softmax
        nnet.set_output_activation_function(Softmax)
        activation_object = nnet.get_output_activation_function()
        self.assertEqual(activation_object, Softmax)


class CreateFromFileTests(unittest.TestCase):
    def make_temp_params_file(self, net_params):
        self.net_params = net_params
        os.makedirs('test_temp', exist_ok=True)
        fname = os.path.join('test_temp', 'neural_net.json')
        self.test_file = fname
        with open(fname, 'w') as f:
            f.write(json.dumps(net_params))

    def setUp(self):
        net_params = {
            'layer_sizes': [1, 2, 2],
            'layers': [{
                'weights': [[3], [-3.5]],
                'biases': [5, -3.1]
            }, {
                'weights': [[1, 0], [-1, -2]],
                'biases': [1, 0]
            }]
        }
        self.net_params = net_params
        self.make_temp_params_file(net_params)

    def tearDown(self):
        os.remove(self.test_file)

    def test_returns_neural_net(self):
        nnet = NeuralNet.create_from_file(fname=self.test_file)
        self.assertIsInstance(nnet, NeuralNet)

    def test_created_net_has_correct_layer_sizes(self):
        nnet = NeuralNet.create_from_file(fname=self.test_file)
        self.assertEqual(nnet.layer_sizes(), self.net_params['layer_sizes'])

        net_params = {
            'layer_sizes': [1, 1, 1],
            'layers': [{
                'weights': [[3]],
                'biases': [5]
            }, {
                'weights': [[1]],
                'biases': [1]
            }]
        }
        self.make_temp_params_file(net_params)
        nnet = NeuralNet.create_from_file(fname=self.test_file)
        self.assertEqual(nnet.layer_sizes(), self.net_params['layer_sizes'])

    def test_created_net_weights(self):
        nnet = NeuralNet.create_from_file(fname=self.test_file)
        weights = nnet.weights()

        expected_weights1 = self.net_params['layers'][0]['weights']
        expected_weights2 = self.net_params['layers'][1]['weights']

        self.assertEqual(weights[0].tolist(), expected_weights1)
        self.assertEqual(weights[1].tolist(), expected_weights2)

    def test_created_net_biases(self):
        nnet = NeuralNet.create_from_file(fname=self.test_file)
        biases = nnet.biases()

        expected_biases1 = self.net_params['layers'][0]['biases']
        expected_biases2 = self.net_params['layers'][1]['biases']

        self.assertEqual(biases[0].tolist(), expected_biases1)
        self.assertEqual(biases[1].tolist(), expected_biases2)

    def test_for_net_with_multiple_hidden_layers(self):
        net_params = {
            'layer_sizes': [1, 1, 1, 2],
            'layers': [{
                'weights': [[3]],
                'biases': [5]
            }, {
                'weights': [[1]],
                'biases': [1]
            }, {
                'weights': [[1.5], [3]],
                'biases': [10, 20]
            }]
        }
        self.make_temp_params_file(net_params)
        nnet = NeuralNet.create_from_file(fname=self.test_file)
        weights = nnet.weights()
        biases = nnet.biases()

        self.assertEqual(nnet.layer_sizes(), self.net_params['layer_sizes'])

        expected_weights1 = self.net_params['layers'][0]['weights']
        expected_weights2 = self.net_params['layers'][1]['weights']
        expected_weights3 = self.net_params['layers'][2]['weights']

        self.assertEqual(weights[0].tolist(), expected_weights1)
        self.assertEqual(weights[1].tolist(), expected_weights2)
        self.assertEqual(weights[2].tolist(), expected_weights3)

        expected_biases1 = self.net_params['layers'][0]['biases']
        expected_biases2 = self.net_params['layers'][1]['biases']
        expected_biases3 = self.net_params['layers'][2]['biases']

        self.assertEqual(biases[0].tolist(), expected_biases1)
        self.assertEqual(biases[1].tolist(), expected_biases2)
        self.assertEqual(biases[2].tolist(), expected_biases3)


class SaveTests(unittest.TestCase):
    def setUp(self):
        self.dest_fname = os.path.join('test_temp', 'nets_params.json')

    def tearDown(self):
        os.remove(self.dest_fname)

    def test_creates_file(self):
        nnet = NeuralNet(layer_sizes=[2, 1, 2])
        dest_fname = os.path.join('test_temp', 'nets_params.json')
        nnet.save(dest_fname=dest_fname)
        self.assertTrue(os.path.isfile(dest_fname))

    def test_json_is_valid(self):
        nnet = NeuralNet(layer_sizes=[2, 1, 2])
        nnet.randomize_parameters()
        nnet.save(self.dest_fname)
        with open(self.dest_fname, 'r') as f:
            s = f.read()
            try:
                json.loads(s)
            except:
                self.assertTrue(False, 'Invalid json')

    def test_json_structure_is_correct(self):
        nnet = NeuralNet(layer_sizes=[2, 1, 2])
        nnet.save(self.dest_fname)
        with open(self.dest_fname, 'r') as f:
            net_params = json.loads(f.read())
            self.assertIn('layer_sizes', net_params)
            self.assertIn('layers', net_params)
            self.assertIsInstance(net_params['layers'], list)
            self.assertIn('weights', net_params['layers'][0])
            self.assertIn('biases', net_params['layers'][0])

    def test_correct_parameters(self):
        nnet = NeuralNet(layer_sizes=[2, 1, 2])
        nnet.randomize_parameters()
        nnet.save(self.dest_fname)
        with open(self.dest_fname, 'r') as f:
            net_params = json.loads(f.read())
            self.assertEqual(net_params['layer_sizes'], [2, 1, 2])
            self.assertEqual(net_params['layers'][0]['weights'], nnet.weights()[0].tolist())
            self.assertEqual(net_params['layers'][1]['weights'], nnet.weights()[1].tolist())

            self.assertEqual(net_params['layers'][0]['biases'], nnet.biases()[0].tolist())
            self.assertEqual(net_params['layers'][1]['biases'], nnet.biases()[1].tolist())

    def test_costs_match(self):
        nnet = NeuralNet(layer_sizes=[2, 3, 1, 10, 2])
        nnet.randomize_parameters()

        X = [np.array([90, 23], float), np.array([0, 2], float)]
        Y = [np.array([0.4, 0.6], float), np.array([0.3, 0])]
        examples = [X, Y]
        c1 = nnet.get_cost(examples)

        fname = os.path.join('test_temp', 'nets_params.json')
        nnet.save(dest_fname=fname)

        nnet = NeuralNet.create_from_file(fname)
        c2 = nnet.get_cost(examples)
        self.assertAlmostEqual(c1, c2, places=4)
