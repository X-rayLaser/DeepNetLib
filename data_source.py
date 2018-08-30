import helpers


class DataSource:
    def next_batch(self, batch_size):
        raise Exception('Not implemented')

    def end_of_data(self):
        raise Exception('Not implemented')

    def restart(self):
        raise Exception('Not implemented')


class PreloadSource(DataSource):
    def __init__(self, examples, shuffled=False):
        if shuffled:
            x_list, y_list = helpers.shuffle_pairwise(*examples)
        else:
            x_list, y_list = examples

        self._index = 0
        self._xlist = x_list
        self._ylist = y_list

        assert len(self._xlist) == len(self._ylist)

    def next_batch(self, size):
        index = self._index
        xs = self._xlist[index:index + size]
        ys = self._ylist[index:index + size]
        self._index += size
        return xs, ys

    def end_of_data(self):
        return self._index >= len(self._ylist)

    def restart(self):
        self._index = 0


class ExampleIterarator:
    def __init__(self, data_source):
        self._src = data_source

    def __iter__(self):
        return self

    def __next__(self):
        if self.end_of_data():
            raise StopIteration()

        return self._src.next_batch(size=1)[0]

    def end_of_data(self):
        return self._src.end_of_data()

    def next(self):
        return self.__next__()


class MiniBatchIterator(ExampleIterarator):
    def __init__(self, data_source, batch_size=50):
        ExampleIterarator.__init__(self, data_source=data_source)
        self._size = batch_size

    def __next__(self):
        if self.end_of_data():
            raise StopIteration()
        examples = self._src.next_batch(batch_size=self._size)
        child_src = PreloadSource(examples=examples)
        return ExampleIterarator(data_source=child_src)
