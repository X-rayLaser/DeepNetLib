import numpy as np
import math


def quadratic_per_example(activation, expected_output):
    v = activation - expected_output
    return v.dot(v) / 2.0


def quadratic_cost(activations, outputs):
    vector_len = len(activations)

    s = 0
    for i in range(vector_len):
        s += quadratic_per_example(activation=activations[i],
                                   expected_output=outputs[i])
    return s / vector_len


def cross_entropy_between_numbers(a, y):
    """Cross entropy between 2 numbers, a and y defined on [0, 1]"""
    if a == 0 and y == 0:
        return 0

    if a == 1 and y == 1:
        return 0

    if a == 0 or a == 1:
        return np.finfo(float).max

    return -y * math.log(a) - (1 - y) * math.log(1 - a, math.e)


def cross_entropy_per_example(a, y):
    ncomponents = a.shape[0]
    ce = 0
    for i in range(ncomponents):
        ce += cross_entropy_between_numbers(a[i], y[i])

    return ce


def cross_entropy(activations, outputs):
    ce = 0
    num_of_examples = len(outputs)
    for i in range(num_of_examples):
        ce += cross_entropy_per_example(activations[i], outputs[i])
    return ce / num_of_examples


class QuadraticCost:
    def __init__(self):
        pass

    def compute_cost(self, activations, outputs):
        return quadratic_cost(activations=activations, outputs=outputs)


class CrossEntropyCost:
    def compute_cost(self, activations, outputs):
        return cross_entropy(activations=activations, outputs=outputs)
