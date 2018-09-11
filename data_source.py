import helpers


class DataSource:
    def get_examples(self, index_from, index_to):
        raise NotImplementedError()

    def number_of_examples(self):
        raise NotImplementedError()


class PreloadSource(DataSource):
    def __init__(self, examples, shuffled=False):
        if shuffled:
            x_list, y_list = helpers.shuffle_pairwise(*examples)
        else:
            x_list, y_list = examples

        self._xlist = x_list
        self._ylist = y_list

        assert len(self._xlist) == len(self._ylist)

    def get_examples(self, index_from, index_to):
        i = index_from
        j = index_to
        return self._xlist[i:j], self._ylist[i:j]

    def number_of_examples(self):
        return len(self._ylist)


class DataSetIterator:
    def __init__(self, data_source):
        self._src = data_source
        self._index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.end_of_data():
            raise StopIteration()

        i = self._index
        X, Y = self._src.get_examples(i, i+1)
        self._index += 1
        return X[0], Y[0]

    def end_of_data(self):
        return self._index >= self._src.number_of_examples()

    def next(self):
        return self.__next__()


class BatchesIterator(DataSetIterator):
    def __init__(self, data_source, batch_size=50):
        DataSetIterator.__init__(self, data_source=data_source)
        self._batch_size = batch_size
        self._index = 0

    def __next__(self):
        if self.end_of_data():
            raise StopIteration()

        i = self._index
        examples = self._src.get_examples(i, i + self._batch_size)
        self._index += self._batch_size
        child_src = PreloadSource(examples=examples)
        return child_src
