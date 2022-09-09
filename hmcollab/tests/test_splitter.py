import unittest

import datetime
import numpy as np

from hmcollab import datasets
from hmcollab import directories
from hmcollab import splitter


class TestSplitter(unittest.TestCase):
    def setUp(self):
        self.tree = datasets.HMDatasetDirectoryTree(
            base=directories.testdata("forty_more_customers")
        )
        self.dataset = datasets.HMDataset(tree=self.tree)

    def tearDown(self):
        pass

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

    def test_split_by_customer_train_test_strategy(self):
        strategy = splitter.SplitByCustomerTrainTestStrategy(
            self.dataset, 0.1, random_state=np.random.RandomState(48)
        )
        ids_test = strategy.test.customers.customer_id.unique()
        actual = set(ids_test)
        expected = {
            "00005ca1c9ed5f5146b52ac8639a40ca9d57aeff4d1bd2c5feb1ca5dff07c43e",
            "d3b658f59ad9bbf6249a7bf0db722f2f43cc47803d03299a3feb24487f1b6fbe",
            "00007d2de826758b65a93dd24ce629ed66842531df6699338c5570910a014cc2",
            "df6ae5106281e6a424ca88bf70cf9a0fdbbea20669e44a6f93ac7ed5cb1863e5",
            "0000423b00ade91418cceaf3b26c6af3dd342b51fd051eec9c12fb36984420fa",
            "00008469a21b50b3d147c97135e25b4201a8c58997f78782a0cc706645e14493",
        }
        self.assertEqual(expected, actual)

        strategy = splitter.SplitByCustomerTrainTestStrategy(
            self.dataset, 0.1, random_state=np.random.RandomState(42)
        )
        ids_test = strategy.test.customers.customer_id.unique()
        actual = set(ids_test)
        self.assertNotEqual(expected, actual)

    def test_split_by_customer_train_test_validation_strategy_default(self):
        strategy = splitter.SplitByCustomerTrainTestValidationStrategy(
            self.dataset, 0.2, 0.2
        )

        # make sure dataframes in train, test and validation have correct shapes
        for part in strategy.partition:
            self.assertEqual(self.dataset.customers.shape[1], part.customers.shape[1])
            self.assertEqual(self.dataset.articles.shape[1], part.articles.shape[1])
            self.assertEqual(
                self.dataset.transactions.shape[1], part.transactions.shape[1]
            )

        expected = self.dataset.customers.shape[0]
        actual = sum([part.customers.shape[0] for part in strategy.partition])
        self.assertEqual(expected, actual)

        expected = self.dataset.transactions.shape[0]
        actual = sum([part.transactions.shape[0] for part in strategy.partition])
        self.assertEqual(expected, actual)

        # make sure that customer_ids are same in all partitions
        for part in strategy.partition:
            expected = set(part.customers.customer_id.unique())
            actual = set(part.transactions.customer_id.unique())
            self.assertEqual(expected, actual)

        train_ids = set(strategy.train.customers.customer_id.unique())
        test_ids = set(strategy.test.customers.customer_id.unique())
        valid_ids = set(strategy.validation.customers.customer_id.unique())

        self.assertEqual(0, len(train_ids.intersection(test_ids)))
        self.assertEqual(0, len(train_ids.intersection(valid_ids)))
        self.assertEqual(0, len(valid_ids.intersection(test_ids)))

        full_ids = set(self.dataset.customers.customer_id.unique())

        union = set(train_ids).union(test_ids).union(valid_ids)
        self.assertEqual(full_ids, union)

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
        actual = test_ids
        self.assertEqual(expected, actual)

        expected = {
            "11b511d20267f66ab323cd61141a7ed797399d45c67ef151971737dab9a93d36",
            "000058a12d5b43e67d225668fa1f8d618c13dc232df0cad8ffe7ad4a1091e318",
            "725b95bd9404996b4a2c9b7f6e2dd3fd19cbe8c0a919cf93b0606cccb245f034",
            "ecec63e8840393e2087453b86c5967c2e8f9ddf66d07348881206d45afdc65e9",
            "cb37880afc6777d7c63ecc851086aadc3ed06b5bec1975ea892430f3013c792c",
            "0df50cbaccd0bbd3ffca435197a84998d5334202f2676f67728b132da984b49d",
            "d529e1c4120cda88d7c3906926ef6f73c3ea3114605ab9fc89dc1e61904fb1a1",
            "00007e8d4e54114b5b2a9b51586325a8d0fa74ea23ef77334eaec4ffccd7ebcc",
            "00007d2de826758b65a93dd24ce629ed66842531df6699338c5570910a014cc2",
            "0000757967448a6cb83efb3ea7a3fb9d418ac7adf2379d8cd0c725276a467a2a",
            "d1c91a9050fcb6da04f0643c6b6d38b002b6ff12d66e804de258f69825b02fca",
        }
        actual = valid_ids
        self.assertEqual(expected, actual)

    def test_split_by_customer_train_test_validation_strategy(self):
        strategy = splitter.SplitByCustomerTrainTestValidationStrategy(
            self.dataset, 0.1, 0.05, random_state=np.random.RandomState(48)
        )
        ids_test = strategy.test.customers.customer_id.unique()
        actual = set(ids_test)
        expected = {
            "00005ca1c9ed5f5146b52ac8639a40ca9d57aeff4d1bd2c5feb1ca5dff07c43e",
            "d3b658f59ad9bbf6249a7bf0db722f2f43cc47803d03299a3feb24487f1b6fbe",
            "00007d2de826758b65a93dd24ce629ed66842531df6699338c5570910a014cc2",
            "df6ae5106281e6a424ca88bf70cf9a0fdbbea20669e44a6f93ac7ed5cb1863e5",
            "0000423b00ade91418cceaf3b26c6af3dd342b51fd051eec9c12fb36984420fa",
            "00008469a21b50b3d147c97135e25b4201a8c58997f78782a0cc706645e14493",
        }
        self.assertEqual(expected, actual)

        ids_valid = strategy.validation.customers.customer_id.unique()
        actual = set(ids_valid)
        expected = {
            "c020c2dfb076bd6192e555684a86cd3bc1cd0d913fa8aff151118b36a5db11ef",
            "309de68e2003d3944ce891fa38f1f7b190522eb6b63146a0bff941fb8b339be5",
            "9bf81bb97882f398974e2d0190d253f6eb218f3c2103ab30d290d36a078d1b75",
        }
        self.assertEqual(expected, actual)
