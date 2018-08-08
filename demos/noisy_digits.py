import sys
import os
import numpy as np

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from digit_drawing import DigitGenerator
import helpers


def noisy_digit_vector(digit, mu=0, std=0.2):
    size = 10
    v = helpers.category_to_vector(digit, size)
    g = np.random.normal(mu, std, size)
    return g + v


def draw_noisy(digit, trials):
    for t in range(trials):
        pixels = gen.generate(seeding_vector=noisy_digit_vector(digit))

        fname = os.path.join(dest_folder, 'digit_{}_{}th.png'.format(digit, t))

        helpers.create_image(dest_fname=fname, pixel_vector=pixels,
                             width=image_width, height=image_height)
        print('Generated {}th image of a digit {}'.format(t, digit))


dest_folder = 'generated_digits'
image_width = 28
image_height = 28
nepochs = 10

gen = DigitGenerator()

helpers.download_dataset()
pixels_to_categories = helpers.get_training_data()
gen.train(pixels_to_categories=pixels_to_categories, nepochs=nepochs)

print('Training for {} epochs is complete'.format(nepochs))


for i in range(10):
    draw_noisy(digit=i, trials=50)
