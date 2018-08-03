import numpy as np


def sigma(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigma_prime(z):
    return sigma(z) * (1 - sigma(z))
