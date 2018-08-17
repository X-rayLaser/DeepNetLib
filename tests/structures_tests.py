import unittest
from structures import LinkedList


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
