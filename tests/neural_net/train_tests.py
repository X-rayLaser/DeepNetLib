import unittest
import numpy as np
from main import NeuralNet, NetFactory
import helpers


class NeuralNetTrain(unittest.TestCase):
    def test_gives_correct_output_on_training_data(self):
        nnet = NetFactory.create_neural_net(sizes=[1, 1, 1])

        xes = [np.array([-10], float), np.array([100], float)]
        ys = [np.array([0.5], float), np.array([0.75], float)]

        nnet.train(examples=(xes, ys), nepochs=100)

        for i in range(len(xes)):
            res = nnet.feed(xes[i])
            self.assertAlmostEqual(res[0], ys[i][0], places=1)

    def test_gives_correct_output_for_unseen_data(self):
        nnet = NetFactory.create_neural_net(sizes=[1, 10, 1])

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