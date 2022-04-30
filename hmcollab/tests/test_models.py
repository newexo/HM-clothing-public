import unittest

from hmcollab import datasets
from hmcollab import directories
from hmcollab import articles
from hmcollab import models


class TestModels(unittest.TestCase):
    def setUp(self):
        self.tree = datasets.HMDatasetDirectoryTree(base=directories.testdata("ten_customers"))
        self.dataset = datasets.HMDataset(tree=self.tree)
        self.articles_munger = articles.ArticleFeaturesSimpleFeatures(
            self.dataset.articles, use_article_id=True
        )
        self.customer = (
            "0000757967448a6cb83efb3ea7a3fb9d418ac7adf2379d8cd0c725276a467a2a"
        )
        self.clusters = 2
        self.days = 7
        self.full_dummies = articles.ArticleFeaturesSimpleFeatures(
            self.dataset.articles, use_article_id=True
        ).x

        self.customer_list = [
            "000058a12d5b43e67d225668fa1f8d618c13dc232df0cad8ffe7ad4a1091e318",
            "0000757967448a6cb83efb3ea7a3fb9d418ac7adf2379d8cd0c725276a467a2a",
        ]

    def tearDown(self):
        pass

    def test_knn_recommender(self):
        recommender = models.KnnRecommender(
            self.dataset,
            self.full_dummies,
            groups=2,
            total_recommendations=6,
        )
        actual = recommender.recommend(self.customer)
        expected = ['0715624008', '0783388001', '0377277001', '0726925001', '0735843004', '0559630026']
        self.assertEqual(expected, actual)

    def test_recommend_all(self):
        recommender = models.KnnRecommender(
            self.dataset,
            self.full_dummies,
            groups=2,
            total_recommendations=4,
        )
        df = recommender.recommend_all(self.customer_list)

        actual = df.shape
        expected = (2, 2)
        self.assertEqual(expected, actual)

        actual = len(df.prediction[0].split(' '))
        expected = 4
        self.assertEqual(expected, actual)

        actual = df.prediction[0]

        expected = '0351484002 0663713001 0870304002 0578020002'
        self.assertEqual(expected, actual)

        actual = df.prediction[1]
        expected = '0715624008 0783388001 0726925001 0735843004'
        self.assertEqual(expected, actual)
