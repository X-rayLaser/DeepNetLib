import unittest
import numpy as np
import helpers
import backprop_slow
from main import NeuralNet


class BackpropSlowTests(unittest.TestCase):
    def compare_grads(self, grad1, grad2):
        self.assertTrue(helpers.gradients_equal(grad1, grad2))

    def test_back_propagation_slow(self):
        nnet = NeuralNet(layer_sizes=[1, 1, 1])
        x = np.array([5], float)
        y = np.array([0.25], float)
        examples = ([x], [y])
        w_grad, b_grad = backprop_slow.back_propagation_slow(examples=examples, neural_net=nnet)

        w_grad_expected = [np.array([[0]], float), np.array([[1/32]], float)]
        b_grad_expected = [np.array([[0]], float), np.array([[1/16]], float)]

        self.compare_grads(w_grad, w_grad_expected)
        self.compare_grads(b_grad, b_grad_expected)

    def test_back_propagation_slow_type_array(self):
        nnet = NeuralNet(layer_sizes=[2, 1, 2])
        x = np.array([5, 2], float)
        y = np.array([0.25, 0], float)

        examples = ([x], [y])
        w_grad, b_grad = backprop_slow.back_propagation_slow(examples=examples, neural_net=nnet)
        self.assertIsInstance(w_grad, list)
        self.assertIsInstance(w_grad[0], np.ndarray)
        self.assertIsInstance(w_grad[1], np.ndarray)

        self.assertIsInstance(b_grad, list)
        self.assertIsInstance(b_grad[0], np.ndarray)
        self.assertIsInstance(b_grad[1], np.ndarray)

    def test_back_propagation_slow_shape(self):
        nnet = NeuralNet(layer_sizes=[3, 2, 2, 5])
        x = np.array([5, 2, -0.5], float)
        y = np.array([0.25, 0, 0, 0.7, 0.2], float)
        examples = ([x], [y])
        w_grad, b_grad = backprop_slow.back_propagation_slow(examples=examples, neural_net=nnet)
        self.assertEqual(len(w_grad), 3)
        self.assertEqual(len(b_grad), 3)
        self.assertTupleEqual(w_grad[0].shape, (2, 3))
        self.assertTupleEqual(w_grad[1].shape, (2, 2))
        self.assertTupleEqual(w_grad[2].shape, (5, 2))

        self.assertTupleEqual(b_grad[0].shape, (2,))
        self.assertTupleEqual(b_grad[1].shape, (2,))
        self.assertTupleEqual(b_grad[2].shape, (5,))


from backprop import LinkedList


class LinkedListTests(unittest.TestCase):
    def test_empty(self):
        mylist = LinkedList()
        self.assertRaises(LinkedList.EndOfListError, lambda: mylist.tail())
        self.assertRaises(LinkedList.EndOfListError, lambda: mylist.get_item())

        self.assertTrue(mylist.is_empty())

    def test_with_one_element(self):
        mylist = LinkedList()
        s = 'hello'
        mylist.prepend(s)
        self.assertFalse(mylist.is_empty())
        item = mylist.get_item()
        self.assertEqual(item, s)
        newlist = mylist.tail()
        self.assertTrue(newlist.is_empty())

    def test_recurrent_tail(self):
        mylist = LinkedList()
        mylist.prepend(1)
        mylist.prepend(2)
        mylist.prepend(3)
        self.assertEqual(mylist.get_item(), 3)

        mylist = mylist.tail()
        self.assertEqual(mylist.get_item(), 2)

        mylist = mylist.tail()
        self.assertEqual(mylist.get_item(), 1)
        mylist = mylist.tail()
        self.assertTrue(mylist.is_empty())

    def test_to_pylist(self):
        mylist = LinkedList()
        mylist.prepend(11)
        mylist.prepend(22)
        mylist.prepend(33)
        pylist = mylist.to_pylist()
        self.assertEqual(pylist, [33, 22, 11])

        mylist = LinkedList()
        mylist.prepend('s')
        pylist = mylist.to_pylist()
        self.assertEqual(pylist, ['s'])
