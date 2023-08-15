import unittest

from hmcollab.directory_tree import HMDatasetDirectoryTree
from hmcollab import datasets
from hmcollab import directories
from hmcollab import articles
from hmcollab import models
from hmcollab import scoring

import warnings
import pandas as pd

warnings.filterwarnings("ignore")


# TODO: replace these tests with integration tests depending on data in testdata/fivehundred
class TestModels(unittest.TestCase):
    def setUp(self):
        self.tree = HMDatasetDirectoryTree(base=directories.testdata("ten_customers"))
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
            threshold=0,
        )
        actual = recommender.recommend(self.customer)
        expected = [
            "0726925001",
            "0735843004",
            "0559630026",
            "0715624008",
            "0783388001",
            "0377277001",
        ]
        self.assertEqual(expected, actual)

    def test_recommend_all(self):
        recommender = models.KnnRecommender(
            self.dataset,
            self.full_dummies,
            groups=2,
            total_recommendations=4,
            threshold=0,
        )
        df = recommender.recommend_all(self.customer_list, drop_duplicates=False)

        actual = df.shape
        expected = (2, 2)
        self.assertEqual(expected, actual)

        actual = len(df.prediction[0].split(" "))
        expected = 4
        self.assertEqual(expected, actual)

        actual = df.prediction[0]

        expected = "0351484002 0663713001 0870304002 0578020002"
        self.assertEqual(expected, actual)

        actual = df.prediction[1]
        expected = "0726925001 0735843004 0715624008 0783388001"
        self.assertEqual(expected, actual)

    def test_recommend_all_drop_duplicates(self):
        recommender = models.KnnRecommender(
            self.dataset,
            self.full_dummies,
            groups=2,
            total_recommendations=4,
            threshold=0,
        )
        df = recommender.recommend_all(self.customer_list, drop_duplicates=True)

        actual = df.shape
        expected = (2, 2)
        self.assertEqual(expected, actual)

        actual = len(df.prediction[0].split(" "))
        expected = 4
        self.assertEqual(expected, actual)

        actual = df.prediction[0]

        expected = "0351484002 0723529001 0727808001 0852643001"
        self.assertEqual(expected, actual)

        actual = df.prediction[1]
        expected = "0726925001 0735843004 0715624008 0783388001"
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

        expected = "0351484002 0723529001 0351484002 0723529001"
        self.assertEqual(expected, actual)

        actual = df.prediction[1]
        expected = "0351484002 0723529001 0351484002 0723529001"
        self.assertEqual(expected, actual)

    def test_popular_recommender(self):
        recommender = models.PopularRecommender(
            self.dataset,
            self.full_dummies,
            total_recommendations=12,
        )

        # 1. Test recommend
        actual_recommend = recommender.recommend()
        expected = [
            "0351484002",
            "0723529001",
            "0811835004",
            "0689898002",
            "0640174001",
            "0797065001",
            "0599580055",
            "0811927004",
            "0811925005",
            "0800436010",
            "0666448006",
            "0663713001",
        ]

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

        y_by_customer = pd.read_csv(
            directories.data(filename="target_set_7d_75481u.csv")
        )

        # originally, prediction failed for all. Fix so that one recommendation is correct to test ap_at_k and map_at_k
        actual_recommend_all.prediction = actual_recommend_all.iloc[
            0
        ].prediction.replace("0811835004", "0794321007")
        t = scoring.relevant(actual_recommend_all, y_by_customer)
        self.assertEqual((1,), t.shape)
        self.assertEqual((12,), t[0].shape)
        self.assertEqual(
            [
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
            list(t[0]),
        )

        self.assertAlmostEquals(0.027777777777777776, scoring.ap_at_k(t[0]))
        self.assertAlmostEquals(0.027777777777777776, scoring.map_at_k(t))
