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
        expected_pixels.fill(127)
        self.assertEqual(pixels.tolist(), expected_pixels.tolist())

    def test_returns_non_uniform_array_after_training(self):
        x = [np.array([0]*784, float)]
        y = [np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], float)]
        pixels_to_categories = (x, y)

        diggen = DigitGenerator()
        diggen.train(pixels_to_categories=pixels_to_categories)
        pixels = diggen.generate_digit(digit=1)
        expected_pixels = np.zeros(784, dtype=np.uint8)
        expected_pixels.fill(127)
        self.assertNotEqual(pixels.tolist(), expected_pixels.tolist())


class ConstructorTests(unittest.TestCase):
    def test_neural_net_has_correct_layer_sizes(self):
        gen = DigitGenerator()
        sizes = gen.net_layer_sizes()
        self.assertEqual(len(sizes), 3)
        self.assertEqual(sizes[0], 10)
        self.assertEqual(sizes[-1], 784)


class PrepareExamplesTests(unittest.TestCase):
    def test_for_single_example(self):
        x = [np.array([9, 124, 203], float)]
        y = [np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], float)]
        pixels_to_categories = (x, y)
        X, Y = DigitGenerator.prepare_train_examples(pixels_to_categories)
        self.assertEqual(X[0].tolist(), y[0].tolist())
        self.assertEqual(Y[0].tolist(), [9 / 255.0, 124 / 255.0, 203 / 255.0])

        x = [np.array([55, 2], float)]
        y = [np.array([0.1, 0.8, 0, 0, 0, 0, 0, 0, 0, 0], float)]
        pixels_to_categories = (x, y)
        X, Y = DigitGenerator.prepare_train_examples(pixels_to_categories)
        self.assertEqual(X[0].tolist(), y[0].tolist())
        self.assertEqual(Y[0].tolist(), [55 / 255.0, 2 / 255.0])

    def test_for_multiple_examples(self):
        x = [np.array([9, 124, 203], float),
             np.array([55, 2], float)]
        y = [np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], float),
             np.array([0.1, 0.8, 0, 0, 0, 0, 0, 0, 0, 0], float)]
        pixels_to_categories = (x, y)
        X, Y = DigitGenerator.prepare_train_examples(pixels_to_categories)
        self.assertEqual(len(X), 2)

        self.assertEqual(X[0].tolist(), y[0].tolist())
        self.assertEqual(Y[0].tolist(), [9 / 255.0, 124 / 255.0, 203 / 255.0])

        self.assertEqual(X[1].tolist(), y[1].tolist())
        self.assertEqual(Y[1].tolist(), [55 / 255.0, 2 / 255.0])
