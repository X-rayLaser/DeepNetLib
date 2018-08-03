import numpy as np
import random


def generate_data(f, start_value, end_value, step_value):
    x = []
    y = []

    farg = start_value
    while farg < end_value:
        v = np.zeros((1,), float)
        v[0] = f(farg)
        x.append(np.array([farg], float))
        y.append(v)
        farg += step_value
    return x, y


def random_input_vector(size):
    sr = random.SystemRandom()
    v = np.zeros((size,), float)
    for i in range(size):
        v[i] = sr.random() * 100 - 50
    return v


def random_output_vector(size):
    sr = random.SystemRandom()
    v = np.zeros((size,), float)
    for i in range(size):
        v[i] = sr.random()
    return v


def generate_random_examples(examples_number, input_size, output_size):
    sr = random.SystemRandom()
    xes = []
    ys = []
    for i in range(examples_number):
        x = random_input_vector(input_size)
        y = random_output_vector(output_size)
        xes.append(x)
        ys.append(y)

    return xes, ys


def gradients_equal(grad1, grad2):
    nmatrices = len(grad1)

    if nmatrices != len(grad2):
        return False

    for i in range(nmatrices):
        g1 = grad1[i]
        g2 = grad2[i]
        mtx = g1 - g2
        s = np.abs(mtx).sum()
        if s > 0.001:
            return False
    return True


def shuffle_pairwise(list_a, list_b):
    a_shuf = []
    b_shuf = []

    shuffled_indices = [i for i in range(len(list_a))]
    random.shuffle(shuffled_indices)
    for i in shuffled_indices:
        a_shuf.append(list_a[i])
        b_shuf.append(list_b[i])

    return a_shuf, b_shuf
