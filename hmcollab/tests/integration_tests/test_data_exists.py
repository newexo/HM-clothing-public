import unittest
import os

from hmcollab import directories
from hmcollab.directory_tree import HMDatasetDirectoryTree


class TestDataExists(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_full_data_exists(self):
        os.path.exists(directories.data("articles.csv"))
        os.path.exists(directories.data("customers.csv"))
        os.path.exists(directories.data("transactions_train.csv"))

    def test_test_data_exists(self):
        tree = HMDatasetDirectoryTree(base=directories.testdata())
        os.path.exists(tree.articles)
        os.path.exists(tree.customers)
        os.path.exists(tree.transactions)
