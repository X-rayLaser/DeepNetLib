import numpy as np
import random
import main


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


def zero_gradients_list(neural_net):
    weights_grad = []
    biases_grad = []
    wlist = neural_net.weights()
    blist = neural_net.biases()

    for i in range(len(wlist)):
        wshape = wlist[i].shape
        weights_grad.append(np.zeros(wshape))

        bshape = blist[i].shape
        biases_grad.append(np.zeros(bshape))

    return weights_grad, biases_grad


def update_total_gradients(summed_gradients_list, new_gradients_list):
    summed_len = len(summed_gradients_list)
    new_len = len(new_gradients_list)
    assert summed_len == new_len

    res_list = []
    for i in range(summed_len):
        res_list.append(summed_gradients_list[i] + new_gradients_list[i])
    return res_list


def average_gradient(gradient_sum, examples_count):
    res_list = []
    for i in range(len(gradient_sum)):
        res_list.append(gradient_sum[i] / float(examples_count))

    return res_list
