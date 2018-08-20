import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from digit_drawing import DigitGenerator
import helpers
from datasets import mnist


dest_folder = 'generated_digits'
image_width = 28
image_height = 28
nepochs = 5

gen = DigitGenerator()

mnist.download_dataset()
pixels_to_categories = mnist.get_training_data()
gen.train(pixels_to_categories=pixels_to_categories, nepochs=nepochs)

print('Training for {} epochs is complete'.format(nepochs))

for i in range(10):
    pixels = gen.generate_digit(i)
    helpers.create_image(dest_fname=os.path.join(dest_folder, 'digit_{}.png'.format(i)),
                         pixel_vector=pixels, width=image_width, height=image_height)
    print('Generated image of a digit {}'.format(i))
