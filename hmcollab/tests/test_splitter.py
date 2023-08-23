import datetime
import unittest

import numpy as np

from hmcollab import datasets
from hmcollab import directories
from hmcollab import splitter
from hmcollab.directory_tree import HMDatasetDirectoryTree
from hmcollab.three_part_dataset import prune_articles, prune_customers


# TODO: replace these tests with unit tests
# Many of these tests depend only on the fields by which we are splitting and no other fields
class TestSplitter(unittest.TestCase):
    def setUp(self):
        self.tree = HMDatasetDirectoryTree(
            base=directories.testdata("forty_more_customers")
        )
        self.dataset = datasets.HMDataset(tree=self.tree)
        self.larger_tree = HMDatasetDirectoryTree(
            base=directories.testdata("larger_dataset")
        )
        self.larger_dataset = datasets.HMDataset(tree=self.larger_tree)

    def tearDown(self):
        pass

    # TODO: replace with a unit test based on a random dataframe
    def test_split_by_time(self):
        expected = self.dataset.transactions.iloc[0].t_dat
        older, newer = splitter.split_by_time(self.dataset.transactions, days=20)
        actual = self.dataset.transactions.iloc[0].t_dat
        self.assertEqual(expected, actual)

        self.assertEqual(
            self.dataset.transactions.shape[0], older.shape[0] + newer.shape[0]
        )
        new_min = newer.t_dat.min()
        new_max = newer.t_dat.max()
        old_min = older.t_dat.min()
        old_max = older.t_dat.max()

        self.assertGreater(new_min, old_max)

        self.assertEqual(new_max, self.dataset.transactions.t_dat.max())
        self.assertEqual(old_min, self.dataset.transactions.t_dat.min())

        cutoff_date = new_max - datetime.timedelta(days=20)
        self.assertLess(old_max, cutoff_date)

    # TODO: replace with a unit test based on a random dataframe
    def test_older_newer_portions_transactions(self):
        op = splitter.OlderPortion(20).split(self.dataset)
        np = splitter.NewerPortion(20).split(self.dataset)
        older = op.transactions
        newer = np.transactions

        self.assertEqual(
            self.dataset.transactions.shape[0], older.shape[0] + newer.shape[0]
        )
        new_min = newer.t_dat.min()
        new_max = newer.t_dat.max()
        old_min = older.t_dat.min()
        old_max = older.t_dat.max()

        self.assertGreater(new_min, old_max)

        self.assertEqual(new_max, self.dataset.transactions.t_dat.max())
        self.assertEqual(old_min, self.dataset.transactions.t_dat.min())

        cutoff_date = new_max - datetime.timedelta(days=20)
        self.assertLess(old_max, cutoff_date)

    # TODO: replace with a unit test based on a random dataframe
    def test_prune_customers(self):
        customer_ids = {
            "bed6979e4c947ad451fad7235ffab1e61ad95517e8cf26cb9e9d68a4decc9b5f",
            "00005ca1c9ed5f5146b52ac8639a40ca9d57aeff4d1bd2c5feb1ca5dff07c43e",
            "d3b658f59ad9bbf6249a7bf0db722f2f43cc47803d03299a3feb24487f1b6fbe",
        }
        pruned_customers = prune_customers(
            self.dataset.customers, customer_ids=customer_ids
        )
        expected = customer_ids
        actual = set(pruned_customers.customer_id.unique())
        self.assertEqual(expected, actual)

    # TODO: replace with a unit test based on a random dataframe, also consider that this may be covered
    def test_prune_articles(self):
        article_ids = {
            "0456163085",
            "0456163086",
            "0768847001",
            "0568601043",
        }
        pruned_articles = prune_articles(self.dataset.articles, article_ids=article_ids)
        expected = article_ids
        actual = set(pruned_articles.article_id.unique())
        self.assertEqual(expected, actual)

    # TODO: replace with a unit test based on a random dataframe
    def test_split_by_ids(self):
        full_ids = self.dataset.customers.customer_id.unique()
        full_ids = set(full_ids)
        ids_train, ids_test = splitter.split_ids(self.dataset.customers, 0.2)
        self.assertEqual(len(full_ids), len(ids_train) + len(ids_test))
        actual = set(ids_test)
        expected = {
            "bed6979e4c947ad451fad7235ffab1e61ad95517e8cf26cb9e9d68a4decc9b5f",
            "00005ca1c9ed5f5146b52ac8639a40ca9d57aeff4d1bd2c5feb1ca5dff07c43e",
            "d3b658f59ad9bbf6249a7bf0db722f2f43cc47803d03299a3feb24487f1b6fbe",
            "c020c2dfb076bd6192e555684a86cd3bc1cd0d913fa8aff151118b36a5db11ef",
            "0650bd95fe9658f9afa9a21fe08823fa41de1fb11f6f819a655c16ef1ba51e7b",
            "000064249685c11552da43ef22a5030f35a147f723d5b02ddd9fd22452b1f5a6",
            "18cfbd899a5f5f3b4bc0a0430104e0fd436c9fb10402eb184d95582a9f59d8b3",
            "08104b0ff2b1f4422fcf1547ddc175bdc17f9a6cec37a3f1f07d5a13ca1168a9",
            "309de68e2003d3944ce891fa38f1f7b190522eb6b63146a0bff941fb8b339be5",
            "7f52372053f4282d546a2b8eb4dfdbf17a5df363e2825a87c6c8a2843872fbda",
            "e9d5c82ff885d03aa0c2a69db4c98cf95675c59a0c5634624eaa593e23db8973",
        }
        self.assertEqual(expected, actual)
        union = actual.union(ids_train)
        self.assertEqual(full_ids, union)

    # TODO: replace with a unit test based on a random dataframe
    def test_transactions_train_test(self):
        ids_train, ids_test = splitter.split_ids(self.dataset.customers, 0.2)
        train, test = splitter.transactions_train_test(
            self.dataset.transactions, ids_train, ids_test
        )
        self.assertEqual(
            self.dataset.transactions.shape[0], train.shape[0] + test.shape[0]
        )

        expected = set(ids_test)
        actual = set(test.customer_id.unique())
        self.assertEqual(expected, actual)

        expected = set(ids_train)
        actual = set(train.customer_id.unique())
        self.assertEqual(expected, actual)

    # TODO: replace with a simpler test
    def test_customer_portion(self):
        all_customer_ids = self.dataset.customers.customer_id.unique()
        r = np.random.RandomState(42)
        customer_ids = r.choice(all_customer_ids, size=4, replace=False)
        cp = splitter.CustomerPortion(customer_ids)
        ds = cp.split(self.dataset)

        actual = ds.customers.shape
        expected = (4, 7)
        self.assertEqual(expected, actual)

        actual = ds.transactions.shape
        expected = (319, 5)
        self.assertEqual(expected, actual)

        actual = ds.articles.shape
        expected = (264, 25)
        self.assertEqual(expected, actual)

        expected = set(customer_ids)
        actual = set(ds.customers.customer_id.unique())
        self.assertEqual(expected, actual)

        actual = set(ds.transactions.customer_id.unique())
        self.assertEqual(expected, actual)

        expected = set(ds.articles.article_id.unique())
        actual = set(ds.transactions.article_id.unique())
        self.assertEqual(expected, actual)

    # TODO: consider replacing with a simpler test
    def test_split_by_customer_train_test_strategy_default(self):
        strategy = splitter.SplitByCustomerTrainTestStrategy(self.dataset, 0.2)
        full_ids = self.dataset.customers.customer_id.unique()
        full_ids = set(full_ids)

        # test that test and train customers have expected shapes
        expected = self.dataset.customers.shape[0]
        actual = strategy.train.customers.shape[0] + strategy.test.customers.shape[0]
        self.assertEqual(expected, actual)

        expected = self.dataset.customers.shape[1]
        actual = strategy.train.customers.shape[1]
        self.assertEqual(expected, actual)

        actual = strategy.test.customers.shape[1]
        self.assertEqual(expected, actual)

        # test that test and train articles have expected shapes
        expected = self.dataset.articles.shape[1]
        actual = strategy.train.articles.shape[1]
        self.assertEqual(expected, actual)

        actual = strategy.test.articles.shape[1]
        self.assertEqual(expected, actual)

        # test that test and train transactions have expected shapes
        expected = self.dataset.transactions.shape[0]
        actual = (
            strategy.train.transactions.shape[0] + strategy.test.transactions.shape[0]
        )
        self.assertEqual(expected, actual)

        expected = self.dataset.transactions.shape[1]
        actual = strategy.train.transactions.shape[1]
        self.assertEqual(expected, actual)

        actual = strategy.test.transactions.shape[1]
        self.assertEqual(expected, actual)

        ids_train = strategy.train.customers.customer_id.unique()
        ids_test = strategy.test.customers.customer_id.unique()
        actual = set(ids_test)
        expected = {
            "bed6979e4c947ad451fad7235ffab1e61ad95517e8cf26cb9e9d68a4decc9b5f",
            "00005ca1c9ed5f5146b52ac8639a40ca9d57aeff4d1bd2c5feb1ca5dff07c43e",
            "d3b658f59ad9bbf6249a7bf0db722f2f43cc47803d03299a3feb24487f1b6fbe",
            "c020c2dfb076bd6192e555684a86cd3bc1cd0d913fa8aff151118b36a5db11ef",
            "0650bd95fe9658f9afa9a21fe08823fa41de1fb11f6f819a655c16ef1ba51e7b",
            "000064249685c11552da43ef22a5030f35a147f723d5b02ddd9fd22452b1f5a6",
            "18cfbd899a5f5f3b4bc0a0430104e0fd436c9fb10402eb184d95582a9f59d8b3",
            "08104b0ff2b1f4422fcf1547ddc175bdc17f9a6cec37a3f1f07d5a13ca1168a9",
            "309de68e2003d3944ce891fa38f1f7b190522eb6b63146a0bff941fb8b339be5",
            "7f52372053f4282d546a2b8eb4dfdbf17a5df363e2825a87c6c8a2843872fbda",
            "e9d5c82ff885d03aa0c2a69db4c98cf95675c59a0c5634624eaa593e23db8973",
        }
        self.assertEqual(expected, actual)
        union = actual.union(ids_train)
        self.assertEqual(full_ids, union)

        ids_train = set(strategy.train.transactions.customer_id.unique())
        ids_test = set(strategy.test.transactions.customer_id.unique())
        self.assertEqual(0, len(ids_train.intersection(ids_test)))
        actual = set(ids_test)
        self.assertEqual(expected, actual)
        union = actual.union(ids_train)
        self.assertEqual(full_ids, union)

        full_ids = self.dataset.articles.article_id.unique()
        full_ids = set(full_ids)
        ids_train = strategy.train.articles.article_id.unique()
        ids_test = strategy.test.articles.article_id.unique()
        union = set(ids_test).union(ids_train)
        self.assertEqual(full_ids, union)
