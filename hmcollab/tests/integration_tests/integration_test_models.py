import unittest
import warnings

from hmcollab import articles
from hmcollab import datasets
from hmcollab import directories
from hmcollab import models
from hmcollab.directory_tree import HMDatasetDirectoryTree

warnings.filterwarnings("ignore")


class IntegrationTestModels(unittest.TestCase):
    def setUp(self):
        self.tree = HMDatasetDirectoryTree(base=directories.testdata("fivehundred"))
        self.dataset = datasets.HMDataset(tree=self.tree)
        self.articles_munger = articles.ArticleFeaturesSimpleFeatures(
            self.dataset.articles, use_article_id=True
        )
        self.customer = self.dataset.customers.customer_id[10]
        self.clusters = 2
        self.days = 7
        self.full_dummies = articles.ArticleFeaturesSimpleFeatures(
            self.dataset.articles, use_article_id=True
        ).x

        self.customer_list = [
            self.customer,
            self.dataset.customers.customer_id[100],
        ]

    def tearDown(self):
        pass

    def test_knn_recommender(self):
        recommender = models.KnnRecommender(
            self.dataset,
            self.full_dummies,
            groups=2,
            total_recommendations=6,
            threshold=0,
        )
        actual = recommender.recommend(self.customer)
        expected_indices = [478, 1038, 1309, 427, 7409,  432]
        expected = list(recommender.filtered_dummies.iloc[expected_indices].article_id)
        self.assertEqual(expected, actual)

    def test_recommend_all_drop_duplicates_with_threshold(self):
        recommender = models.KnnRecommender(
            self.dataset,
            self.full_dummies,
            groups=2,
            total_recommendations=4,
            threshold=2,
        )
        df = recommender.recommend_all(self.customer_list, drop_duplicates=True)

        actual = df.shape
        expected = (2, 2)
        self.assertEqual(expected, actual)

        actual = len(df.prediction[0].split(" "))
        expected = 4
        self.assertEqual(expected, actual)

        actual = df.prediction[0]
        expected_indices = [201, 349, 106, 1172]
        expected_list = list(recommender.filtered_dummies.iloc[expected_indices].article_id)
        expected = " ".join(expected_list)
        self.assertEqual(expected, actual)

        actual = df.prediction[1]
        expected_indices = [163, 586, 722, 727]
        expected_list = list(recommender.filtered_dummies.iloc[expected_indices].article_id)
        expected = " ".join(expected_list)
        self.assertEqual(expected, actual)

    def test_popular_recommender(self):
        recommender = models.PopularRecommender(
            self.dataset,
            self.full_dummies,
            total_recommendations=12,
        )

        # 1. Test recommend
        actual_recommend = recommender.recommend()
        expected_indices = [6721, 9356, 2150, 1720, 27, 6722, 9901, 13439, 5507, 2655, 489, 2148]
        expected = list(self.dataset.articles.iloc[expected_indices].article_id)

        self.assertEqual(expected, actual_recommend)

        actual_recommend_all = recommender.recommend_all(self.customer_list)

        actual_rows = actual_recommend_all.shape[0]
        expected = len(self.customer_list)
        self.assertEqual(expected, actual_rows)

        actual = actual_recommend_all.prediction.value_counts().shape[0]
        expected = 1
        self.assertEqual(expected, actual)

        actual = actual_recommend_all.iloc[0, 1]
        expected = " ".join(actual_recommend)
        self.assertEqual(expected, actual)
