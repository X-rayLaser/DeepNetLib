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


def cross_entropy(activations, outputs):
    assert False, 'Not implemented'
