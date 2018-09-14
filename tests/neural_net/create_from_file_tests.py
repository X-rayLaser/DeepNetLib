import unittest
import json
import os
from neural_net import NeuralNet


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
