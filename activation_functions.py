import numpy as np


def sigma(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigma_prime(z):
    return sigma(z) * (1 - sigma(z))


class Rectifier:
    @staticmethod
    def activation(z_vector):
        return np.maximum(np.zeros(z_vector.shape, dtype=float), z_vector)

    @staticmethod
    def gradient(z_vector):
        def f(z):
            if z <= 0:
                return 0
            return 1

        vf = np.vectorize(f)
        return vf(z_vector)
