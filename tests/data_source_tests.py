from unittest import TestCase
import data_source


class PreloadSourceTests(TestCase):
    def test_next_batch_with_size_1(self):
        xs = [0, 1]
        ys = [3, 4]
        src = data_source.PreloadSource(examples=(xs, ys), shuffled=False)

        x, y = src.next_batch(size=1)
        self.assertEquals(x, [0])
        self.assertEquals(y, [3])

        x, y = src.next_batch(size=1)
        self.assertEquals(x, [1])
        self.assertEquals(y, [4])

    def test_next_batch_with_varying_size(self):
        xs = [0, 1]
        ys = [3, 4]
        src = data_source.PreloadSource(examples=(xs, ys), shuffled=False)
        x, y = src.next_batch(size=2)
        self.assertEquals(x, [0, 1])
        self.assertEquals(y, [3, 4])

        xs = [0]
        ys = [3]
        src = data_source.PreloadSource(examples=(xs, ys), shuffled=False)
        x, y = src.next_batch(size=3)
        self.assertEquals(x, [0])
        self.assertEquals(y, [3])

    def test_end_of_data_for_empty_source(self):
        src = data_source.PreloadSource(examples=([], []), shuffled=False)
        self.assertTrue(src.end_of_data())

    def test_end_of_data_for_single_example_source(self):
        src = data_source.PreloadSource(examples=([3], [5]), shuffled=False)
        self.assertFalse(src.end_of_data())

        src.next_batch(size=1)
        self.assertTrue(src.end_of_data())

    def test_restart(self):
        src = data_source.PreloadSource(examples=([3], [5]), shuffled=False)
        src.next_batch(size=1)
        src.restart()
        self.assertFalse(src.end_of_data())

        x, y = src.next_batch(size=3)
        self.assertTupleEqual((x, y), ([3], [5]))
