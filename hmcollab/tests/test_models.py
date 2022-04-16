import unittest

from hmcollab import datasets
from hmcollab import directories
from hmcollab import articles
from hmcollab import models


class TestModels(unittest.TestCase):
    def setUp(self):
        self.tree = datasets.HMDatasetDirectoryTree(base=directories.testdata())
        self.dataset = datasets.HMDataset(tree=self.tree)
        self.articles_munger = articles.ArticleFeaturesSimpleFeatures(
            self.dataset.articles, use_article_id=True
        )
        self.customer = (
            "0000423b00ade91418cceaf3b26c6af3dd342b51fd051eec9c12fb36984420fa"
        )
        self.clusters = 2
        self.days = 7
        self.full_dummies = articles.ArticleFeaturesSimpleFeatures(
            self.dataset.articles, use_article_id=True
        ).x

        self.recommendations = [
            "0795440001",
            "0796137001",
            "0673677002",
            "0590928022",
            "0599580049",
            "0559616014",
        ]

    def tearDown(self):
        pass

    def test_recommend_by_customer(self):
        actual = models.recommender_by_customer(
            self.customer,
            self.dataset,
            self.full_dummies,
            groups=2,
            total_recommendations=6,
        )
        expected = self.recommendations
        self.assertEqual(expected, actual)

    def test_knn_recommender(self):
        recommender = models.KnnRecommender(
            self.dataset,
            self.full_dummies,
            groups=2,
            total_recommendations=6,
        )
        actual = recommender.recommend(self.customer)
        expected = self.recommendations
        self.assertEqual(expected, actual)
