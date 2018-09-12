from unittest import TestCase
import data_source


class ExampleIteratorTests(TestCase):
    def test_next(self):
        xs = [0, 1]
        ys = [3, 4]
        src = data_source.PreloadSource(examples=(xs, ys), shuffled=False)
        it = data_source.DataSetIterator(data_source=src)

        x, y = it.next()
        self.assertEqual(x, 0)
        self.assertEqual(y, 3)

        x, y = it.next()
        self.assertEqual(x, 1)
        self.assertEqual(y, 4)

    def test_next_raises_stop_iteration(self):
        xs = [0]
        ys = [3]

        src = data_source.PreloadSource(examples=(xs, ys), shuffled=False)
        it = data_source.DataSetIterator(data_source=src)

        for t in it:
            self.assertTupleEqual(t, (0, 3))


class MiniBatchIteratorTests(TestCase):
    def test_next_with_batch_size_1(self):
        xs = [0, 1]
        ys = [3, 4]
        src = data_source.PreloadSource(examples=(xs, ys), shuffled=False)
        iter = data_source.BatchesIterator(data_source=src, batch_size=1)
        new_src = iter.next()

        for x, y in data_source.DataSetIterator(new_src):
            self.assertEqual(x, 0)
            self.assertEqual(y, 3)

    def test_next_with_batch_size_2(self):
        xs = [0, 1]
        ys = [3, 4]
        src = data_source.PreloadSource(examples=(xs, ys), shuffled=False)
        iter = data_source.BatchesIterator(data_source=src, batch_size=2)
        new_src = iter.next()
        batch = data_source.DataSetIterator(new_src)
        x, y = batch.next()
        self.assertEqual(x, 0)
        self.assertEqual(y, 3)
        x, y = batch.next()
        self.assertEqual(x, 1)
        self.assertEqual(y, 4)
