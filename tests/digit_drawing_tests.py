import unittest
import numpy as np
from digit_drawing import DigitGenerator


class GenerateDigitTests(unittest.TestCase):
    def test_method_outputs_numpy_array_of_integers(self):
        diggen = DigitGenerator()
        pixels = diggen.generate_digit(digit=5)
        self.assertIsInstance(pixels, np.ndarray)
        self.assertEqual(pixels.dtype,np.uint8)

    def test_returned_array_has_valid_shape(self):
        diggen = DigitGenerator()
        pixels = diggen.generate_digit(digit=1)
        self.assertTupleEqual(pixels.shape, (784,))

    def test_can_only_generate_digits_from_0_to_9(self):
        diggen = DigitGenerator()
        self.assertRaises(DigitGenerator.InvalidDigitError,
                          lambda: diggen.generate_digit(digit=-1))
        self.assertRaises(DigitGenerator.InvalidDigitError,
                          lambda: diggen.generate_digit(digit=-5))
        self.assertRaises(DigitGenerator.InvalidDigitError,
                          lambda: diggen.generate_digit(digit=-50))

        self.assertRaises(DigitGenerator.InvalidDigitError,
                          lambda: diggen.generate_digit(digit=10))

        self.assertRaises(DigitGenerator.InvalidDigitError,
                          lambda: diggen.generate_digit(digit=50))

    def test_works_for_valid_digits(self):
        diggen = DigitGenerator()
        for digit in range(10):
            diggen.generate_digit(digit=digit)

    def test_argument_must_be_integer(self):
        diggen = DigitGenerator()
        self.assertRaises(DigitGenerator.InvalidDigitError,
                          lambda: diggen.generate_digit(digit=2.5))
        self.assertRaises(DigitGenerator.InvalidDigitError,
                          lambda: diggen.generate_digit(digit=[4]))

    def test_untrained_returns_uniform_array(self):
        diggen = DigitGenerator()
        pixels = diggen.generate_digit(digit=1)
        expected_pixels = np.zeros(784, dtype=np.uint8)
        expected_pixels.fill(128)
        self.assertEqual(pixels.tolist(), expected_pixels.tolist())

    def test_returns_non_uniform_array_after_training(self):
        diggen = DigitGenerator()
        diggen.train()
        pixels = diggen.generate_digit(digit=1)
        expected_pixels = np.zeros(784, dtype=np.uint8)
        expected_pixels.fill(128)
        self.assertNotEqual(pixels.tolist(), expected_pixels.tolist())


class ConstructorTests(unittest.TestCase):
    def test_neural_net_has_correct_layer_sizes(self):
        gen = DigitGenerator()
        sizes = gen.net_layer_sizes()
        self.assertEqual(len(sizes), 3)
        self.assertEqual(sizes[0], 10)
        self.assertEqual(sizes[2], 784)


class TrainTests(unittest.TestCase):
    def test_method_outputs_numpy_array_of_integers(self):
        diggen = DigitGenerator()
        diggen.train()
