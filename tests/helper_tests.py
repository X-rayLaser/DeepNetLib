import unittest
import os
import numpy as np
from PIL import Image
from helpers import shuffle_pairwise, list_to_chunks, InvalidChunkSize
import helpers


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


class CategoryToVectorTests(unittest.TestCase):
    def test_with_invalid_number_of_categories(self):
        self.assertRaises(helpers.InvalidNumberOfCategories,
                          lambda: helpers.category_to_vector(cat_index=1, cat_number=-50))
        self.assertRaises(helpers.InvalidNumberOfCategories,
                          lambda: helpers.category_to_vector(cat_index=1, cat_number=0))

    def test_with_invalid_category_index(self):
        self.assertRaises(helpers.CategoryIndexOutOfBounds,
                          lambda: helpers.category_to_vector(cat_index=5, cat_number=5))

        self.assertRaises(helpers.CategoryIndexOutOfBounds,
                          lambda: helpers.category_to_vector(cat_index=25, cat_number=5))

        self.assertRaises(helpers.CategoryIndexOutOfBounds,
                          lambda: helpers.category_to_vector(cat_index=-5, cat_number=1))

    def test_returns_numpy_array_of_floats(self):
        v = helpers.category_to_vector(2, cat_number=4)
        self.assertIsInstance(v, np.ndarray)
        self.assertTrue(v.dtype in [np.float32, np.float64])

    def test_returns_correct_array(self):
        v = helpers.category_to_vector(0, cat_number=3)
        self.assertEqual(v.tolist(), [1.0, 0, 0])

        v = helpers.category_to_vector(1, cat_number=3)
        self.assertEqual(v.tolist(), [0, 1.0, 0])

        v = helpers.category_to_vector(2, cat_number=3)
        self.assertEqual(v.tolist(), [0, 0, 1])


class MakeDigitImage(unittest.TestCase):
    def test_with_invalid_arguments(self):
        vector = np.array([2, 9, 10, 0, 254, 0], dtype=np.uint8)
        self.assertRaises(
            helpers.WrongImageDimensions,
            lambda: helpers.create_image(dest_fname='digit.png',
                                             pixel_vector=vector, width=1, height=2)
        )

        self.assertRaises(
            helpers.WrongDigitVector,
            lambda: helpers.create_image(dest_fname='digit.png',
                                             pixel_vector=[3, 9], width=1, height=2)
        )

    def test_creates_a_file(self):
        dest_file = os.path.join('generated_digits', 'digit.png')
        vector = np.array([2, 9, 10, 0, 254, 0], dtype=np.uint8)
        helpers.create_image(dest_fname=dest_file,
                                 pixel_vector=vector, width=2, height=3)
        self.assertTrue(os.path.isfile(dest_file))

    def test_created_file_is_correct_image(self):
        dest_file = os.path.join('generated_digits', 'digit.png')
        vector = np.array([2, 9, 10, 0, 254, 0], dtype=np.uint8)
        helpers.create_image(dest_fname=dest_file,
                             pixel_vector=vector, width=2, height=3)

        try:
            im = Image.open(dest_file)
            self.assertEqual(im.width, 2)
            self.assertEqual(im.height, 3)
        except Exception as e:
            print (repr(e))
            assert False
