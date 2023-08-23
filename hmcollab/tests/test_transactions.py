import unittest

import pandas as pd
from sklearn.cluster import KMeans

import hmcollab.splitter
from hmcollab import articles
from hmcollab import datasets
from hmcollab import directories
from hmcollab import transactions
from hmcollab.directory_tree import HMDatasetDirectoryTree


# This suit is data dependent
# + split by time: Could use random dataset with dates to test. 
#               We could simplify test_slit_by_time_first_six(self)
# + creation of customer dummies: could use random set?
# + test kmeans on transaction from single customer. Integration test?

def kmeans_consumer_old(customer, transactions_df, full_articles_dummy, k=1):
    # Note: Do not scale dummy features
    basket = transactions.TransactionsByCustomer(transactions_df).all_article_ids(
        customer
    )
    customer_dummies = full_articles_dummy.merge(
        basket, on="article_id", how="inner"
    ).drop(columns="article_id")
    kmeans = KMeans(
        init="k-means++", n_clusters=k, n_init=10, max_iter=300, random_state=42
    )
    return kmeans.fit(customer_dummies)


class TestTransactions(unittest.TestCase):
    def setUp(self):
        self.tree = hmcollab.directory_tree.HMDatasetDirectoryTree(
            base=directories.testdata()
        )
        self.dataset = datasets.HMDataset(tree=self.tree)
        self.articles_munger = articles.ArticleFeaturesSimpleFeatures(
            self.dataset.articles, use_article_id=True
        )
        self.customer = (
            "08f60b0c07fc14fffc8983aec045c80ede7a419793046375a7ef75b6a18afdf0"
        )
        self.clusters = 2
        self.days = 7

    def tearDown(self):
        pass

    def test_split_by_time(self):
        # so far we are testing the number of rows after the split
        # TODO: expand
        six_transactions = self.dataset.transactions.iloc[:6].copy()
        y, x = hmcollab.splitter.split_by_time(six_transactions, self.days)
        self.assertEqual(3, y.shape[0])
        self.assertEqual(3, x.shape[0])

# TODO: Could use random dataset with dates to test. We could simplify
    def test_slit_by_time_first_six(self):
        six_transactions = self.dataset.transactions.iloc[:6].copy()
        x, y = hmcollab.splitter.split_by_time(six_transactions, self.days)

        # verify that returned dataframes have correct columns
        expected = ["t_dat", "customer_id", "article_id", "price", "sales_channel_id"]
        actual = list(y.columns)
        self.assertEqual(expected, actual)

        actual = list(x.columns)
        self.assertEqual(expected, actual)

        # verify timestamps
        actual = list(y.t_dat)
        expected = [pd.Timestamp(year=2020, month=8, day=19)] * 3
        self.assertEqual(expected, actual)

        actual = list(x.t_dat)
        expected = [pd.Timestamp(year=2018, month=9, day=20)] * 3
        self.assertEqual(expected, actual)

        # verify customer ids
        actual = list(y.customer_id)
        expected = [
            "08f60b0c07fc14fffc8983aec045c80ede7a419793046375a7ef75b6a18afdf0",
            "08f60b0c07fc14fffc8983aec045c80ede7a419793046375a7ef75b6a18afdf0",
            "08f60b0c07fc14fffc8983aec045c80ede7a419793046375a7ef75b6a18afdf0",
        ]
        self.assertEqual(expected, actual)

        actual = list(x.customer_id)
        expected = [
            "08f60b0c07fc14fffc8983aec045c80ede7a419793046375a7ef75b6a18afdf0",
            "18cfbd899a5f5f3b4bc0a0430104e0fd436c9fb10402eb184d95582a9f59d8b3",
            "08f60b0c07fc14fffc8983aec045c80ede7a419793046375a7ef75b6a18afdf0",
        ]
        self.assertEqual(expected, actual)

        # verify article ids
        actual = list(y.article_id)
        expected = ["0907534001", "0871517002", "0877274003"]
        self.assertEqual(expected, actual)

        actual = list(x.article_id)
        expected = ["0110065001", "0111586001", "0531697003"]
        self.assertEqual(expected, actual)

        # verify price
        actual = list(y.price)
        expected = [0.0254067796610168, 0.0254067796610168, 0.0338813559322033]
        self.assertEqual(expected, actual)

        actual = list(x.price)
        expected = [0.022864406779661, 0.0127796610169491, 0.022864406779661]
        self.assertEqual(expected, actual)

        # verify sales channel id
        actual = list(y.sales_channel_id)
        expected = [1, 1, 1]
        self.assertEqual(expected, actual)

        actual = list(x.sales_channel_id)
        expected = [1, 2, 1]
        self.assertEqual(expected, actual)

# convert to unit test using random dataset with article_ids
    def test_all_article_ids(self):
        t = transactions.TransactionsByCustomer(self.dataset.transactions)
        actual = t.all_article_ids(self.customer)
        actual = list(actual)
        expected = [
            "0110065001",
            "0531697003",
            "0907534001",
            "0871517002",
            "0877274003",
        ]
        self.assertEqual(expected, actual)

# TODO: Important test. Think on integration test
    def test_kmeans_consumer_2(self):
        # Note: Do not scale dummy features
        full_dummies = articles.ArticleFeaturesSimpleFeatures(
            self.dataset.articles, use_article_id=True
        ).x
        t = transactions.TransactionsByCustomer(self.dataset.transactions)
        basket = t.all_article_ids(self.customer)
        customer_dummies = full_dummies.merge(
            basket, on="article_id", how="right"
        ).drop(columns="article_id")
        kmeans = KMeans(
            init="k-means++",
            n_clusters=self.clusters,
            n_init=10,
            max_iter=300,
            random_state=42,
        )
        expected = kmeans.fit(customer_dummies).cluster_centers_
        actual = transactions.kmeans_consumer(
            customer_dummies, k=self.clusters
        ).cluster_centers_

        self.assertEqual(expected.tolist(), actual.tolist())
