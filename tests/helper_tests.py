import unittest
import os
import numpy as np
import backprop
from activation_functions import sigma, sigma_prime
from main import NeuralNet
from helpers import shuffle_pairwise, list_to_chunks, InvalidChunkSize, download_dataset, get_training_data, get_test_data


class HelpersTests(unittest.TestCase):
    def test_zero_gradients_list(self):
        nnet = NeuralNet(layer_sizes=[3, 5, 4, 1])
        weights_grads, biases_grads = backprop.zero_gradients_list(neural_net=nnet)
        nmatrices = len(weights_grads)
        self.assertEqual(nmatrices, 3)
        self.assertEqual(nmatrices, len(biases_grads))

        self.assertTupleEqual(weights_grads[0].shape, (5, 3))
        self.assertTupleEqual(weights_grads[1].shape, (4, 5))
        self.assertTupleEqual(weights_grads[2].shape, (1, 4))

        self.assertTupleEqual(biases_grads[0].shape, (5,))
        self.assertTupleEqual(biases_grads[1].shape, (4,))
        self.assertTupleEqual(biases_grads[2].shape, (1,))

        for w in weights_grads:
            self.assertAlmostEqual(w.sum(), 0, places=8)

        for b in biases_grads:
            self.assertAlmostEqual(b.sum(), 0, places=8)

    def test_update_total_gradients(self):
        grad_total = [np.array([[100, 120, 130], [50, 10, 60]], int),
                      np.array([[10, 20], [10, 10]], int)]

        grad_last = [np.array([[5, 10, 20], [10, 10, 10]], int),
                     np.array([[1, 1], [2, 4]], int)]

        grad = backprop.update_total_gradients(summed_gradients_list=grad_total,
                                              new_gradients_list=grad_last)

        expected_grad = [np.array([[105, 130, 150], [60, 20, 70]], int),
                         np.array([[11, 21], [12, 14]], int)]

        self.assertEqual(len(grad), 2)
        self.assertTupleEqual(grad[0].shape, (2, 3))
        self.assertTupleEqual(grad[1].shape, (2, 2))

        self.assertTrue(np.all(grad[0] == expected_grad[0]))
        self.assertTrue(np.all(grad[1] == expected_grad[1]))

    def test_average_gradient(self):
        mtx1 = np.array(
            [[5, 10],
             [2, 4]], float
        )

        mtx2 = np.array([[1, 2, 4]], float)
        gradient_sum = [mtx1, mtx2]

        mtx1_expected = np.array(
            [[1, 2],
             [0.4, 0.8]], float
        )

        mtx2_expected = np.array([[0.2, 0.4, 0.8]], float)

        expected_gradient = [mtx1_expected, mtx2_expected]
        grad = backprop.average_gradient(gradient_sum=gradient_sum, examples_count=5)

        self.assertEqual(len(grad), 2)
        self.assertTupleEqual(grad[0].shape, (2, 2))
        self.assertTupleEqual(grad[1].shape, (1, 3))

        self.assertTrue(np.all(grad[0] == expected_gradient[0]))
        self.assertTrue(np.all(grad[1] == expected_gradient[1]))

    def test_compute_activations_and_zsums(self):
        nnet = NeuralNet(layer_sizes=[2, 3, 2])
        x = np.array([0.5, 3], float)
        nnet.randomize_parameters()

        a, zs = backprop.compute_activations_and_zsums(x=x, neural_net=nnet)
        expected_activations = nnet.feed(x=x)
        self.assertTrue(np.allclose(a[-1], expected_activations))

    def test_compute_errors(self):
        nnet = NeuralNet(layer_sizes=[1, 2, 1])
        cost_func = nnet.get_cost_function()
        nnet.set_weight(layer=1, row=0, col=0, new_value=-0.5)
        nnet.set_weight(layer=1, row=1, col=0, new_value=1.5)
        nnet.set_weight(layer=2, row=0, col=1, new_value=5)

        nnet.set_bias(layer=1, row=0, new_value=1)
        nnet.set_bias(layer=2, row=0, new_value=-1)

        x = np.array([2], float)

        y = np.array([1], float)

        a, zs = backprop.compute_activations_and_zsums(x=x, neural_net=nnet)

        errors_list = backprop.compute_errors(neural_net=nnet, output_activations=a[-1],
                                             expected_output=y, weighed_sums=zs)

        expected_nabla2 = (a[-1] - y) * sigma_prime(zs[-1])
        expected_nabla1 = cost_func.get_error_in_layer(nabla_next=expected_nabla2,
                                                     w_next=np.array([[0, 5]]), z=zs[0])

        self.assertTrue(np.allclose(errors_list[0], expected_nabla1))
        self.assertTrue(np.allclose(errors_list[1], expected_nabla2))


class ShufflePairwiseTests(unittest.TestCase):
    def setUp(self):
        self.list1 = [1, 2, 3, 4, 5]
        self.list2 = [2, 4, 6, 8, 10]

    def test_lenghts_are_preserved(self):
        shuf1, shuf2 = shuffle_pairwise(self.list1, self.list2)
        self.assertEqual(len(shuf1), len(self.list1))
        self.assertEqual(len(shuf2), len(self.list1))

    def test_shuffled_lists_contain_all_original_items(self):
        shuf1, shuf2 = shuffle_pairwise(self.list1, self.list2)
        for i in range(1, 6):
            self.assertIn(i, shuf1)

        for i in range(1, 6):
            self.assertIn(i * 2, shuf2)

    def test_lists_are_shuffled_pairwise(self):
        shuf1, shuf2 = shuffle_pairwise(self.list1, self.list2)
        for i in range(len(shuf1)):
            x = shuf1[i]
            y = shuf2[i]
            self.assertEqual(y, 2 * x)

    def test_original_lists_are_unchanged(self):
        shuffle_pairwise(self.list1, self.list2)
        self.assertSequenceEqual(self.list1, [1, 2, 3, 4, 5])
        self.assertSequenceEqual(self.list2, [2, 4, 6, 8, 10])


class ListToChunksTests(unittest.TestCase):
    def test_keeps_original_list_unmodified(self):
        mylist = [i for i in range(1, 10)]
        chunks = list_to_chunks(mylist, chunk_size=3)
        self.assertSequenceEqual(mylist, [i for i in range(1, 10)])

    def test_correct_sum_of_number_of_elements_in_chunks(self):
        nelem = 10
        mylist = [i for i in range(1, nelem+1)]
        chunks = list_to_chunks(mylist, chunk_size=3)
        counts = [len(chunk) for chunk in chunks]
        self.assertEqual(sum(counts), nelem)

    def test_split_into_1_chunk(self):
        mylist = [1, 2]
        chunks = list_to_chunks(mylist, chunk_size=2)
        self.assertEqual(len(chunks), 1)
        self.assertSequenceEqual(chunks[0], [1, 2])

        chunks = list_to_chunks(mylist, chunk_size=3)
        self.assertEqual(len(chunks), 1)
        self.assertSequenceEqual(chunks[0], [1, 2])

    def test_split_into_2_chunks(self):
        mylist = [1, 2, 3, 4, 5]
        chunks = list_to_chunks(mylist, chunk_size=3)
        self.assertEqual(len(chunks), 2)
        self.assertSequenceEqual(chunks[0], [1, 2, 3])
        self.assertSequenceEqual(chunks[1], [4, 5])

    def test_even_split(self):
        mylist = [1, 2, 3, 4, 5, 6]
        chunks = list_to_chunks(mylist, chunk_size=3)
        self.assertEqual(len(chunks), 2)
        self.assertSequenceEqual(chunks[0], [1, 2, 3])
        self.assertSequenceEqual(chunks[1], [4, 5, 6])

        chunks = list_to_chunks(mylist, chunk_size=2)
        self.assertEqual(len(chunks), 3)
        self.assertSequenceEqual(chunks[0], [1, 2])
        self.assertSequenceEqual(chunks[1], [3, 4])
        self.assertSequenceEqual(chunks[2], [5, 6])

    def test_uneven_split(self):
        mylist = [1, 2, 3, 4, 5]
        chunks = list_to_chunks(mylist, chunk_size=3)
        self.assertEqual(len(chunks), 2)
        self.assertSequenceEqual(chunks[0], [1, 2, 3])
        self.assertSequenceEqual(chunks[1], [4, 5])

        chunks = list_to_chunks(mylist, chunk_size=2)
        self.assertEqual(len(chunks), 3)
        self.assertSequenceEqual(chunks[0], [1, 2])
        self.assertSequenceEqual(chunks[1], [3, 4])
        self.assertSequenceEqual(chunks[2], [5])

    def test_split_empty_list(self):
        chunks = list_to_chunks([], chunk_size=1)
        self.assertEqual(len(chunks), 0)
        chunks = list_to_chunks([], chunk_size=2)
        self.assertEqual(len(chunks), 0)
        chunks = list_to_chunks([], chunk_size=12)
        self.assertEqual(len(chunks), 0)

    def test_split_one_element_list(self):
        chunks = list_to_chunks([5], chunk_size=1)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], [5])

        chunks = list_to_chunks([5], chunk_size=2)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], [5])

        chunks = list_to_chunks([5], chunk_size=20)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], [5])

    def test_split_into_0_size_chunks(self):
        self.assertRaises(InvalidChunkSize, lambda: list_to_chunks([3, 2], chunk_size=0))
        self.assertRaises(InvalidChunkSize, lambda: list_to_chunks([3, 2], chunk_size=-10))


class MNISTLoadingTests(unittest.TestCase):
    def setUp(self):
        os.environ['testing'] = "true"

    def tearDown(self):
        os.remove(os.path.join('examples', 'train-images-idx3-ubyte.gz'))
        os.remove(os.path.join('examples', 'train-labels-idx1-ubyte.gz'))
        os.remove(os.path.join('examples', 't10k-images-idx3-ubyte.gz'))
        os.remove(os.path.join('examples', 't10k-labels-idx1-ubyte.gz'))

        os.remove(os.path.join('examples', 'train-images-idx3-ubyte'))
        os.remove(os.path.join('examples', 'train-labels-idx1-ubyte'))
        os.remove(os.path.join('examples', 't10k-images-idx3-ubyte'))
        os.remove(os.path.join('examples', 't10k-labels-idx1-ubyte'))

    def file_exists(self, fname, expected_size):
        file_path = os.path.join('examples', fname)
        return os.path.isfile(file_path) and os.path.getsize(file_path) == expected_size

    def test_downloading_will_create_train_images_file(self):
        download_dataset()
        self.assertTrue(self.file_exists('train-images-idx3-ubyte.gz', 9912422))

    def test_downloading_will_create_train_labels_file(self):
        download_dataset()
        self.assertTrue(self.file_exists('train-labels-idx1-ubyte.gz', 28881))

    def test_downloading_will_create_test_images_file(self):
        download_dataset()
        self.assertTrue(self.file_exists('t10k-images-idx3-ubyte.gz', 1648877))

    def test_downloading_will_create_test_labels_file(self):
        download_dataset()
        self.assertTrue(self.file_exists('t10k-labels-idx1-ubyte.gz', 4542))

    def test_get_training_data(self):
        download_dataset()
        X_train, Y_train = get_training_data()
        self.assertEqual(len(X_train), 60000)
        self.assertEqual(len(Y_train), 60000)

        self.assertTupleEqual(X_train[0].shape, (28*28, ))
        self.assertTupleEqual(Y_train[0].shape, (10, ))

        self.assertEqual(X_train[0].dtype, 'float64')
        self.assertEqual(Y_train[0].dtype, 'float64')

    def test_get_test_data(self):
        download_dataset()
        X, Y = get_test_data()
        self.assertEqual(len(X), 10000)
        self.assertEqual(len(Y), 10000)

        self.assertTupleEqual(X[0].shape, (28*28, ))
        self.assertTupleEqual(Y[0].shape, (10, ))

        self.assertEqual(X[0].dtype, 'float64')
        self.assertEqual(Y[0].dtype, 'float64')
