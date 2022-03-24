import unittest

import numpy as np
from numpy.linalg import norm

from hmcollab import articles
from hmcollab import datasets
from hmcollab import directories


class TestArticles(unittest.TestCase):
    def setUp(self):
        self.tree = datasets.HMDatasetDirectoryTree(base=directories.testdata())
        self.dataset = datasets.HMDataset(tree=self.tree)

    def tearDown(self):
        pass

    def get_simple(self):
        return articles.ArticleFeaturesSimpleFeatures(self.dataset.articles)

    def test_article_simple_feature_array(self):
        a = self.get_simple()
        expected = (16, 63)
        actual = a.x.shape
        self.assertEqual(expected, actual)

        # number of rows in original dataframe should be same as in matrix representation
        expected, _ = a.df.shape
        actual, _ = a.x.shape
        self.assertEqual(expected, actual)

        # test that actual onehot values are the same as a previously saved example
        expected = np.load(directories.testdata("simple_onehot.npy"))
        actual = a.x.values
        self.assertEqual(0, norm(actual - expected))

    def test_id_from_index(self):
        a = self.get_simple()
        expected = "0111586001"
        actual = a.id_from_index(2)
        self.assertEqual(expected, actual)
