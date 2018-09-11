import unittest
import json
import os
import numpy as np
from main import NeuralNet, NetFactory
from cost_functions import CrossEntropyCost
from data_source import PreloadSource


class SaveTests(unittest.TestCase):
    def setUp(self):
        self.dest_fname = os.path.join('test_temp', 'nets_params.json')

    def tearDown(self):
        os.remove(self.dest_fname)

    def test_creates_file(self):
        nnet = NetFactory.create_neural_net(sizes=[2, 1, 2])
        dest_fname = os.path.join('test_temp', 'nets_params.json')
        nnet.save(dest_fname=dest_fname)
        self.assertTrue(os.path.isfile(dest_fname))

    def test_json_is_valid(self):
        nnet = NetFactory.create_neural_net(sizes=[2, 1, 2])
        nnet.randomize_parameters()
        nnet.save(self.dest_fname)
        with open(self.dest_fname, 'r') as f:
            s = f.read()
            try:
                json.loads(s)
            except:
                self.assertTrue(False, 'Invalid json')

    def test_json_structure_is_correct(self):
        nnet = NetFactory.create_neural_net(sizes=[2, 1, 2])
        nnet.save(self.dest_fname)
        with open(self.dest_fname, 'r') as f:
            net_params = json.loads(f.read())
            self.assertIn('layer_sizes', net_params)
            self.assertIn('layers', net_params)
            self.assertIsInstance(net_params['layers'], list)
            self.assertIn('weights', net_params['layers'][0])
            self.assertIn('biases', net_params['layers'][0])

    def test_correct_parameters(self):
        nnet = NetFactory.create_neural_net(sizes=[2, 1, 2])
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
        nnet = NetFactory.create_neural_net(sizes=[2, 3, 1, 10, 2])
        nnet.randomize_parameters()
        cost_func = CrossEntropyCost(nnet)

        X = [np.array([90, 23], float), np.array([0, 2], float)]
        Y = [np.array([0.4, 0.6], float), np.array([0.3, 0])]
        examples = PreloadSource((X, Y))
        c1 = cost_func.get_cost(examples)

        fname = os.path.join('test_temp', 'nets_params.json')
        nnet.save(dest_fname=fname)

        nnet = NeuralNet.create_from_file(fname)
        c2 = cost_func.get_cost(examples)
        self.assertAlmostEqual(c1, c2, places=4)
