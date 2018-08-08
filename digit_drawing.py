import numpy as np


class DigitGenerator:
    class InvalidDigitError(Exception):
        pass

    def __init__(self):
        pass

    def train(self):
        pass

    def generate(self, seeding_vector):
        pass

    def generate_digit(self, digit):
        if type(digit) != int or digit < 0 or digit > 9:
            raise self.InvalidDigitError(
                'Digit must be an integer from 0 to 9. Got'.format(digit)
            )
        return np.zeros(784, dtype=np.uint8)

    def save_as_json(self, fname):
        pass

    def load_from_json(self, fname):
        pass
