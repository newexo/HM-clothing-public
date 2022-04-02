import unittest

import numpy as np
from numpy.linalg import norm
import pandas as pd

from hmcollab import articles
from hmcollab import datasets
from hmcollab import directories

# just for testing
def dummy_features(df, columns):
    # for articles
    return pd.get_dummies(df[['article_id'] + columns], columns=columns, prefix=columns)


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
        expected = (17, 65)
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

    def test_use_article_ids(self):
        features = [
            "product_type_no",
            "product_group_name",
            "graphical_appearance_no",
            "colour_group_code",
            "perceived_colour_value_id",
            "perceived_colour_master_id",
            "department_no",
            "index_code",
            "index_group_no",
            "section_no",
            "garment_group_no",
        ]
        expected = dummy_features(self.dataset.articles, features)
        munger = articles.ArticleFeaturesSimpleFeatures(self.dataset.articles, use_article_id=True)
        actual = munger.x
        # test that columns are same
        self.assertEqual(list(expected.columns), list(actual.columns))

        # test that dummy values are same
        self.assertEqual(0, norm(expected.values[:, 1:] - actual.values[:, 1:]))

        # test article ids are same
        self.assertEqual(list(expected.article_id), list(actual.article_id))
