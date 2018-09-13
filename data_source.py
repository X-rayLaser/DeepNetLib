import helpers


class DataSource:
    """
    Encapsulates a source of training data for a learning algorithms.
    
    Public abstract methods:
        get_examples: get a tuple of python lists with inputs and outputs 
        number_of_examples: get a total number of training examples
    """
    def get_examples(self, index_from, index_to):
        """
        Grab a batch of training examples from specified range.
        
        :param index_from: index of the first example to be included
        :param index_to: exclude examples starting from this index 
        :return: a tuple of 2 python lists each element of which is numpy 1d array
        """
        raise NotImplementedError()

    def number_of_examples(self):
        """
        
        :return: get a total number of training examples
        """
        raise NotImplementedError()


class PreloadSource(DataSource):
    """
    A subclass of DataSource class.
    
    It is a simple wrapper storing examples in memory.
    
    It is not suitable for working with large training sets.
    """
    def __init__(self, examples, shuffled=False):
        """
        
        :param examples: a tuple of 2 python lists each element of which is numpy 1d array
        :param shuffled: boolean indicating whether random shuffling is needed 
        """
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
    """
    A simple iterator. It returns one example at a time.
    
    Can be used in for loops like an ordinary python collection, e. g.
    for x, y in DataSetIterator(data_source):
        pass
    """
    def __init__(self, data_source):
        """
        
        :param data_source: an instance of DataSource subclass
        """
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
    """
    A subclass of DataSetIterator.
    
    It allows one to iterate over a batches of training examples of fixed size.
    
    Actually, it always returns an instance of DataSource subclass which
    stores the next batch of training examples.
    
    It is convenient to use it to implement a stochastic gradient
    descent algorithm. 
    """
    def __init__(self, data_source, batch_size=50):
        """
        
        :param data_source: an instance of DataSource subclass 
        :param batch_size: a number of training examples per batch, int 
        """
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
