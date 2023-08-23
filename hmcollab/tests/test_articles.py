import unittest

import numpy as np
import pandas as pd
from numpy.linalg import norm

import hmcollab.models
from hmcollab import articles
from hmcollab import datasets
from hmcollab import directories
from hmcollab.directory_tree import HMDatasetDirectoryTree, read_with_article_id


# The suite test the following:
# + articles dataset and preprocessing such as:
#   shape and one-hot encoding implementation
#   Those tests can be replaced with unittest using 
#   One-hot and indeces can be tested with a simple synthetic dataset
#   consisting of a couple of categorical columns and a numerical index
#   column with some IDS starting with 0
# + Other are integration test testing results from KNN
#    We might want to remove those and only keep integration test are models

# just for testing
def dummy_features(df, columns):
    # for articles
    return pd.get_dummies(df[["article_id"] + columns], columns=columns, prefix=columns)


class TestArticles(unittest.TestCase):
    def setUp(self):
        self.tree = HMDatasetDirectoryTree(base=directories.testdata())
        self.dataset = datasets.HMDataset(tree=self.tree)
        self.simple_onehot = np.load(directories.testdata("simple_onehot.npy"))
        one_of_each_dir = directories.testdata("one_of_each")
        self.one_of_each_dummies = read_with_article_id(
            directories.qualifyname(one_of_each_dir, "one_of_each_dummies.csv")
        )
        self.one_of_each_dataset = datasets.HMDataset(
            tree=HMDatasetDirectoryTree(one_of_each_dir)
        )

    def tearDown(self):
        pass

# TODO: Need synthetic data
    def get_simple(self):
        return articles.ArticleFeaturesSimpleFeatures(self.dataset.articles.iloc[:17])

# TODO: Need synthetic data to test .x (dummies). A column with categorial might be enough
    def get_simple_knn(self):
        return hmcollab.models.ArticleKNN(self.get_simple().x, 4)

# TODO: Some duplicated with test_use_article_ids
# Keep but simplify. Convert to unittest (instead of integration)
    def test_article_simple_feature_array(self):
        a = self.get_simple()
        expected = (17, 65)
        actual = a.x.shape
        self.assertEqual(expected, actual)

        # number of rows in original dataframe should be same as in matrix representation
        expected, _ = a.df.shape
        actual, _ = a.x.shape
        self.assertEqual(expected, actual)

# Remove?
        # test that actual onehot values are the same as a previously saved example
        expected = self.simple_onehot
        actual = a.x.values
        self.assertEqual(0, norm(actual - expected))

# TODO Replace using synthetic data with numerical indices 
#       Make sure some of those indices start with 0
    def test_id_from_index(self):
        a = self.get_simple()
        expected = "0111586001"
        actual = a.id_from_index(2)
        self.assertEqual(expected, actual)

# TODO: review if we want to keep this or only integration test at models
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
        expected = {12, 10, 2, 1}
        actual = set(indices[0])
        self.assertEqual(expected, actual)

# TODO: review if we want to keep this or only integration test at models
    def test_knn_by_index(self):
        a = self.get_simple()
        knn = self.get_simple_knn()
        x = self.simple_onehot

        # choose row 10, for which there are two other exact matches
        row = a.x.values[10]

        d, indices = knn.nearest(row)

        # check that actual distances match expected
        expected = np.array([0, 0, 0, 2])
        actual = np.array(d[0])
        self.assertEqual(0, norm(actual - expected))

        # check that actual indices match expected
        expected = {12, 10, 2, 1}
        actual = set(indices[0])
        self.assertEqual(expected, actual)
