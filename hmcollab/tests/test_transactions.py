import unittest

from numpy.linalg import norm

from hmcollab import datasets
from hmcollab import directories
from hmcollab import articles
from hmcollab import transactions


class TestTransactions(unittest.TestCase):
    def setUp(self):
        self.tree = datasets.HMDatasetDirectoryTree(base=directories.testdata())
        self.dataset = datasets.HMDataset(tree=self.tree)
        self.articles_munger = articles.ArticleFeaturesSimpleFeatures(
            self.dataset.articles, use_article_id=True
        )
        self.customer = "08f60b0c07fc14fffc8983aec045c80ede7a419793046375a7ef75b6a18afdf0"

    def tearDown(self):
        pass

    def test_all_article_ids(self):
        t = transactions.TransactionsByCustomer(self.dataset.transactions)
        actual = t.all_article_ids(
            self.customer
        )
        actual = list(actual)
        expected = ["0110065001", "0531697003"]
        self.assertEqual(expected, actual)

    def test_customer_dummies(self):
        t = transactions.TransactionsByCustomer(self.dataset.transactions)
        basket = t.all_article_ids(self.customer)
        full_articles_dummy = self.articles_munger.x
        customer_dummies = full_articles_dummy.merge(
            basket, on="article_id", how="right"
        ).drop(columns="article_id")

        self.assertEqual(len(self.articles_munger.x.columns) - 1, len(customer_dummies.columns))

        actual = t.customer_dummies(self.customer, full_articles_dummy)
        expected = customer_dummies

        # test that columns are same
        self.assertEqual(list(expected.columns), list(actual.columns))

        # test that dummy values are same
        self.assertEqual(0, norm(expected.values - actual.values))
