import unittest
from activation_functions import sigma, sigma_prime


class SigmoidTests(unittest.TestCase):
    def test_sigma(self):
        self.assertAlmostEqual(sigma(0), 0.5, places=2)
        self.assertAlmostEqual(sigma(50), 1, places=2)
        self.assertAlmostEqual(sigma(-50), 0, places=2)

        self.assertAlmostEqual(sigma(1), 0.731, places=2)
        self.assertAlmostEqual(sigma(-1), 0.2689, places=2)

    def test_sigma_prime(self):
        self.assertAlmostEqual(sigma_prime(0), 0.25, places=3)
        self.assertAlmostEqual(sigma_prime(-50), 0, places=3)
        self.assertAlmostEqual(sigma_prime(50), 0, places=3)

        self.assertAlmostEqual(sigma_prime(50), sigma(50) * (1 - sigma(50)), places=3)
