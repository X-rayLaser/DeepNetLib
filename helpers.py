import random
import os
import numpy as np


def generate_data(f, start_value, end_value, step_value):
    x = []
    y = []

    farg = start_value
    while farg < end_value:
        v = np.zeros((1,), float)
        v[0] = f(farg)
        x.append(np.array([farg], float))
        y.append(v)
        farg += step_value
    return x, y


def random_input_vector(size):
    sr = random.SystemRandom()
    v = np.zeros((size,), float)
    for i in range(size):
        v[i] = sr.random() * 100 - 50
    return v


def random_output_vector(size):
    sr = random.SystemRandom()
    v = np.zeros((size,), float)
    for i in range(size):
        v[i] = sr.random()
    return v


def generate_random_examples(examples_number, input_size, output_size):
    sr = random.SystemRandom()
    xes = []
    ys = []
    for i in range(examples_number):
        x = random_input_vector(input_size)
        y = random_output_vector(output_size)
        xes.append(x)
        ys.append(y)

    return xes, ys


def gradients_equal(grad1, grad2):
    nmatrices = len(grad1)

    if nmatrices != len(grad2):
        return False

    for i in range(nmatrices):
        g1 = grad1[i]
        g2 = grad2[i]
        mtx = g1 - g2
        s = np.abs(mtx).sum()
        if s > 0.001:
            return False
    return True


def shuffle_pairwise(list_a, list_b):
    a_shuf = []
    b_shuf = []

    shuffled_indices = [i for i in range(len(list_a))]
    random.shuffle(shuffled_indices)
    for i in shuffled_indices:
        a_shuf.append(list_a[i])
        b_shuf.append(list_b[i])

    return a_shuf, b_shuf


def list_to_chunks(elem_list, chunk_size):
    if chunk_size <= 0:
        raise InvalidChunkSize('chunk_size should be a positive integer')
    list_size = len(elem_list)

    nchunks = list_size // chunk_size
    chunks = []
    for i in range(nchunks):
        index = i * chunk_size
        chunks.append(elem_list[index:index+chunk_size])

    remainder = list_size % chunk_size
    if remainder > 0:
        index = nchunks * chunk_size
        chunks.append(elem_list[index:])

    return chunks


class InvalidChunkSize(Exception):
    pass


def get_examples_from_csv():
    return [], []


def category_to_vector(cat_index, cat_number):
    if cat_number <= 0:
        raise InvalidNumberOfCategories('Number of categories must be positive integer')

    if cat_index >= cat_number or cat_index < 0:
        raise CategoryIndexOutOfBounds('Category index out of bounds: {}'.format(cat_index))

    res = np.zeros(cat_number)
    res[cat_index] = 1.0
    return res


class InvalidNumberOfCategories(Exception):
    pass


class CategoryIndexOutOfBounds(Exception):
    pass


class WrongImageDimensions(Exception):
    pass


class WrongDigitVector(Exception):
    pass


def create_image(dest_fname, pixel_vector, width, height):
    if not isinstance(pixel_vector, np.ndarray) or pixel_vector.dtype != np.uint8:
        raise WrongDigitVector('digit_vector must be of ndarray of numpy.uint8 elements')

    num_of_pixels = pixel_vector.shape[0]

    if num_of_pixels != width * height:
        raise WrongImageDimensions('Total number of pixels must be = width * height')

    dest_fname.split()
    dirname, fname = os.path.split(dest_fname)
    os.makedirs(dirname, exist_ok=True)

    from PIL import Image

    pixels = pixel_vector.reshape((height, width))
    im = Image.fromarray(pixels.astype('uint8'), mode='L')
    im.save(dest_fname)
