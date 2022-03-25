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
        self.simple_onehot = np.load(directories.testdata("simple_onehot.npy"))

    def tearDown(self):
        pass

    def get_simple(self):
        return articles.ArticleFeaturesSimpleFeatures(self.dataset.articles)

    def get_simple_knn(self):
        return  articles.ArticleKNN(self.get_simple(), 4)

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
        expected = self.simple_onehot
        actual = a.x.values
        self.assertEqual(0, norm(actual - expected))

    def test_id_from_index(self):
        a = self.get_simple()
        expected = "0111586001"
        actual = a.id_from_index(2)
        self.assertEqual(expected, actual)

    def test_knn_by_row(self):
        knn = self.get_simple_knn()
        x = self.simple_onehot

        # choose row 10, for which there are two other exact matches
        row = x[10]
        d, indices = knn.nearest(row=row)

        # check that actual distances match expected
        expected = np.array([0, 0, 0, 2])
        actual = np.array(d[0])
        self.assertEqual(0, norm(actual - expected))

        # check that actual indices match expected
        expected = {12, 10,  2,  1}
        actual = set(indices[0])
        self.assertEqual(expected, actual)

    def test_knn_by_index(self):
        knn = self.get_simple_knn()
        x = self.simple_onehot

        # choose row 10, for which there are two other exact matches
        d, indices = knn.nearest(index=10)

        # check that actual distances match expected
        expected = np.array([0, 0, 0, 2])
        actual = np.array(d[0])
        self.assertEqual(0, norm(actual - expected))

        # check that actual indices match expected
        expected = {12, 10,  2,  1}
        actual = set(indices[0])
        self.assertEqual(expected, actual)

    def test_knn_by_id(self):
        knn = self.get_simple_knn()
        x = self.simple_onehot

        # choose row 10, for which there are two other exact matches
        d, indices = knn.nearest(id="0146730001")

        # check that actual distances match expected
        expected = np.array([0, 0, 0, 2])
        actual = np.array(d[0])
        self.assertEqual(0, norm(actual - expected))

        # check that actual indices match expected
        expected = {12, 10,  2,  1}
        actual = set(indices[0])
        self.assertEqual(expected, actual)
