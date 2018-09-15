import sys
import os
import numpy as np

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from data_source import PreloadSource
from neural_net import NetFactory
from activation_functions import Rectifier, Sigmoid
from cost_functions import QuadraticCost
from gradient_descent import GradientDescent


def x_to_y(x):
    return np.array([x ** 2], float)


X = []
Y = []
x = 2
while x < 11:
    X.append(np.array([x], float))
    Y.append(x_to_y(x))
    x += 1


net = NetFactory.create_neural_net([1, 200, 1], hidden_layer_activation=Sigmoid,
                                   output_layer_activation=Rectifier)

net.randomize_parameters()
cost = QuadraticCost(neural_net=net, l2_reg_term=0.0005)
gd = GradientDescent(neural_net=net, cost_function=cost,
                     learning_rate=0.001)

examples = (X, Y)
src = PreloadSource(examples=examples)
for i in range(5000):
    gd.training_epoch(data_src=src)
    print('Epoch {}. Loss is {}'.format(i, cost.get_cost(data_src=src)))

t = 2
while t < 10:
    x = np.array([t], float)
    y_estimate = net.feed(x)
    print('{} -> {}'.format(x, y_estimate))
    t += 0.1

