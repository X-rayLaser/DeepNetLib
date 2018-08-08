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

    def test_argument_must_be_integer(self):
        diggen = DigitGenerator()
        self.assertRaises(DigitGenerator.InvalidDigitError,
                          lambda: diggen.generate_digit(digit=2.5))
