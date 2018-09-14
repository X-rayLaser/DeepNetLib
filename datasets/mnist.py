import os
from shutil import copyfile
import gzip
import numpy as np
import requests
import requests_mock
from mnist import MNIST


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
        src = os.path.join('examples', fname)
        dst = os.path.join('examples', 'cached', fname)
        copyfile(src, dst)


def uncompress(src, dest):
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
    """
    Download a MNIST data set from a website and store it locally.
    :return: None
    """
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
    """
    Get training data of MNIST data set.
    
    :return: a tuple of 2 lists, a list of input vectors and a list of
        corresponding output vectors (both as numpy 1d arrays)
    """
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
    """
    Get test data of MNIST data set.

    :return: a tuple of 2 lists, a list of input vectors and a list of
        corresponding output vectors (both as numpy 1d arrays)
    """
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
