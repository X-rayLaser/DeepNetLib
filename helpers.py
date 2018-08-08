import random
import os
import numpy as np
import requests
import requests_mock
from mnist import MNIST


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


def make_mocked_request(url, target_fname):
    path = cached_file_path(target_fname)
    with open(path, 'rb') as f:
        content = f.read()

    with requests_mock.Mocker() as m:
        m.get(url, content=content)
        return requests.get(url)


def cached_file_path(fname):
    return os.path.join('examples', 'cached', fname)


def cached_exists(fname):
    return os.path.isfile(cached_file_path(fname))


def cache_file(fname):
    if not cached_exists(fname):
        os.makedirs(os.path.join('examples', 'cached'), exist_ok=True)
        from shutil import copyfile
        src = os.path.join('examples', fname)
        dst = os.path.join('examples', 'cached', fname)
        copyfile(src, dst)


def uncompress(src, dest):
    import gzip

    with open(src, 'rb') as fs:
        unzipped_data = gzip.decompress(fs.read())
        with open(dest, 'wb') as fd:
            fd.write(unzipped_data)


def download_file_or_get_cached(url, saved_fname):
    base_url = 'http://yann.lecun.com/exdb/mnist/'
    full_url = base_url + url
    if cached_exists(saved_fname):
        r = make_mocked_request(url=full_url, target_fname=saved_fname)
    else:
        r = requests.get(full_url)

    saved_gzipped_path = os.path.join('examples', saved_fname)
    f = open(saved_gzipped_path, 'wb')
    f.write(r.content)
    f.close()

    cache_file(saved_fname)

    unzipped_fname, gz_extension = os.path.splitext(saved_fname)
    unzipped_path = os.path.join('examples', unzipped_fname)
    uncompress(src=saved_gzipped_path, dest=unzipped_path)


def download_dataset():
    os.makedirs('examples', exist_ok=True)

    download_file_or_get_cached(url='train-images-idx3-ubyte.gz',
                                saved_fname='train-images-idx3-ubyte.gz')

    download_file_or_get_cached(url='train-labels-idx1-ubyte.gz',
                                saved_fname='train-labels-idx1-ubyte.gz')

    download_file_or_get_cached(url='t10k-images-idx3-ubyte.gz',
                                saved_fname='t10k-images-idx3-ubyte.gz')

    download_file_or_get_cached(url='t10k-labels-idx1-ubyte.gz',
                                saved_fname='t10k-labels-idx1-ubyte.gz')


def get_training_data():
    mndata = MNIST('examples')

    X_train = []
    Y_train = []

    images, labels = mndata.load_training()

    for image in images:
        X_train.append(np.array(image, float))

    for label in labels:
        y = np.zeros((10, ), float)
        y[label] = 1.0
        Y_train.append(y)

    return X_train, Y_train


def get_test_data():
    mndata = MNIST('examples')

    X = []
    Y = []

    images, labels = mndata.load_testing()

    for image in images:
        X.append(np.array(image, float))

    for label in labels:
        y = np.zeros((10,), float)
        y[label] = 1.0
        Y.append(y)

    return X, Y


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

