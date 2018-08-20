import math
import numpy as np


class Sigmoid:
    @staticmethod
    def activation(z):
        """
        Apply element-wise a sigmoid function to a numpy array z
        
        :param z: number or numpy 1d array
        :return: number or numpy 1d array
        """
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def gradient(z):
        """
        Get a vector of partial derivatives of the sigmoid with respect to each element of z
        :param z: number or numpy 1d array
        :return: number or numpy 1d array
        """
        return Sigmoid.activation(z) * (1 - Sigmoid.activation(z))


class Rectifier:
    @staticmethod
    def activation(z_vector):
        """
        Apply element-wise a rectifier function to a numpy array z

        :param z_vector: number or numpy 1d array
        :return: number or numpy 1d array
        """
        return np.maximum(np.zeros(z_vector.shape, dtype=float), z_vector)

    @staticmethod
    def gradient(z_vector):
        def f(z):
            if z <= 0:
                return 0
            return 1

        vf = np.vectorize(f)
        return vf(z_vector)


class Softmax:
    @staticmethod
    def activation(z_vector):
        denominator = 0
        for z in z_vector:
            denominator += math.exp(z)

        def f(n):
            return math.exp(n) / denominator

        vf = np.vectorize(f)
        a = vf(z_vector)
        return a

    @staticmethod
    def gradient(z_vector):
        vectorlen = z_vector.shape[0]
        jacobian = np.zeros((vectorlen, vectorlen))

        softmax = Softmax.activation(z_vector)

        for i in range(vectorlen):
            for j in range(vectorlen):
                si = softmax[i]
                sj = softmax[j]
                if i == j:
                    jacobian[i, j] = si * (1 - sj)
                else:
                    jacobian[i, j] = si * sj

        return jacobian
