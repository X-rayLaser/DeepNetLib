import unittest
import os
from datasets.mnist import download_dataset, get_training_data, get_test_data


class MNISTLoadingTests(unittest.TestCase):
    def setUp(self):
        os.environ['testing'] = "true"

    def tearDown(self):
        try:
            os.remove(os.path.join('examples', 'train-images-idx3-ubyte.gz'))
            os.remove(os.path.join('examples', 'train-labels-idx1-ubyte.gz'))
            os.remove(os.path.join('examples', 't10k-images-idx3-ubyte.gz'))
            os.remove(os.path.join('examples', 't10k-labels-idx1-ubyte.gz'))

            os.remove(os.path.join('examples', 'train-images-idx3-ubyte'))
            os.remove(os.path.join('examples', 'train-labels-idx1-ubyte'))
            os.remove(os.path.join('examples', 't10k-images-idx3-ubyte'))
            os.remove(os.path.join('examples', 't10k-labels-idx1-ubyte'))
        except Exception as e:
            print(repr(e))

    def file_exists(self, fname, expected_size):
        file_path = os.path.join('examples', fname)
        return os.path.isfile(file_path) and os.path.getsize(file_path) == expected_size

    def test_downloading_will_create_necessary_files(self):
        if not os.environ.get('run_slow', None):
            return
        download_dataset()
        self.assertTrue(self.file_exists('train-images-idx3-ubyte.gz', 9912422))
        self.assertTrue(self.file_exists('train-labels-idx1-ubyte.gz', 28881))
        self.assertTrue(self.file_exists('t10k-images-idx3-ubyte.gz', 1648877))
        self.assertTrue(self.file_exists('t10k-labels-idx1-ubyte.gz', 4542))

    def test_get_training_data(self):
        if not os.environ.get('run_slow', None):
            return
        download_dataset()
        X_train, Y_train = get_training_data()
        self.assertEqual(len(X_train), 60000)
        self.assertEqual(len(Y_train), 60000)

        self.assertTupleEqual(X_train[0].shape, (28*28, ))
        self.assertTupleEqual(Y_train[0].shape, (10, ))

        self.assertEqual(X_train[0].dtype, 'float64')
        self.assertEqual(Y_train[0].dtype, 'float64')

    def test_get_test_data(self):
        if not os.environ.get('run_slow', None):
            return

        download_dataset()
        X, Y = get_test_data()
        self.assertEqual(len(X), 10000)
        self.assertEqual(len(Y), 10000)

        self.assertTupleEqual(X[0].shape, (28*28, ))
        self.assertTupleEqual(Y[0].shape, (10, ))

        self.assertEqual(X[0].dtype, 'float64')
        self.assertEqual(Y[0].dtype, 'float64')