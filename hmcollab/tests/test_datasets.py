import unittest
import os
import pandas as pd

from hmcollab import datasets
from hmcollab import directories
from hmcollab import transactions


class TestDatasets(unittest.TestCase):
    def setUp(self):
        self.tree = datasets.HMDatasetDirectoryTree(base=directories.testdata())
        self.dataset = datasets.HMDataset(tree=self.tree)
        self.x_y_r = datasets.Target(self.dataset.transactions)
        # self.toy = datasets.HMDataset(toy=True)

    def tearDown(self):
        pass

    def test_directory_tree(self):
        tree = datasets.HMDatasetDirectoryTree(base="base")
        expected = "base"
        actual = tree.path()
        self.assertEqual(expected, actual)

        expected = "base/offset"
        actual = tree.path("offset")
        self.assertEqual(expected, actual)

        expected = 'base/articles.csv'
        actual = tree.articles
        self.assertEqual(expected, actual)

        expected = 'base/customers.csv'
        actual = tree.customers
        self.assertEqual(expected, actual)

        expected = 'base/transactions_train.csv'
        actual = tree.transactions
        self.assertEqual(expected, actual)

        expected = 'base/images/123/1234567890.jpg'
        actual = tree.image(1234567890)
        self.assertEqual(expected, actual)

        expected = directories.data()
        actual = datasets.HMDatasetDirectoryTree().path()
        self.assertEqual(expected, actual)

    def test_trees_directories_exist(self):
        self.assertTrue(os.path.isdir(self.tree.path()))
        self.assertTrue(os.path.isdir(self.tree.images()))

    def test_tree_files_exist(self):
        self.assertTrue(os.path.exists(self.tree.articles))
        self.assertTrue(os.path.exists(self.tree.customers))
        self.assertTrue(os.path.exists(self.tree.transactions))
        self.assertTrue(os.path.exists(self.tree.image("1234567890")))
        self.assertTrue(os.path.exists(self.tree.image("5678901234")))

    def test_articles(self):
        expected = ['article_id', 'product_code', 'prod_name', 'product_type_no',
                    'product_type_name', 'product_group_name', 'graphical_appearance_no',
                    'graphical_appearance_name', 'colour_group_code', 'colour_group_name',
                    'perceived_colour_value_id', 'perceived_colour_value_name',
                    'perceived_colour_master_id', 'perceived_colour_master_name',
                    'department_no', 'department_name', 'index_code', 'index_name',
                    'index_group_no', 'index_group_name', 'section_no', 'section_name',
                    'garment_group_no', 'garment_group_name', 'detail_desc']
        actual = list(self.dataset.articles.columns)
        self.assertEqual(expected, actual)
        self.assertEqual(110, self.dataset.articles.shape[0])

    def test_article_id(self):
        expected = "0110065001"
        actual = self.dataset.articles.iloc[0].article_id
        self.assertEqual(expected, actual)

    def test_transaction_article_id(self):
        expected = "0110065001"
        actual = self.dataset.transactions.iloc[0].article_id
        self.assertEqual(expected, actual)

    def test_product_code(self):
        expected = "0110065"
        actual = self.dataset.articles.iloc[0].product_code
        self.assertEqual(expected, actual)

    def test_colour_group_code(self):
        expected = "09"
        actual = self.dataset.articles.iloc[0].colour_group_code
        self.assertEqual(expected, actual)

    def test_customers(self):
        expected = ['customer_id', 'FN', 'Active', 'club_member_status',
                    'fashion_news_frequency', 'age', 'postal_code']
        actual = list(self.dataset.customers.columns)
        self.assertEqual(expected, actual)
        self.assertEqual(5, self.dataset.customers.shape[0])

    def test_transactions(self):
        expected = ['t_dat', 'customer_id', 'article_id', 'price', 'sales_channel_id']
        actual = list(self.dataset.transactions.columns)
        self.assertEqual(expected, actual)
        self.assertEqual(120, self.dataset.transactions.shape[0])

    def test_target_vs_last7d(self):
        # Target dataset should have column target rather than column last7d
        df = pd.read_csv(directories.data("target_set_7d_75481u.csv"), nrows=20)
        self.assertNotIn("last_7d", df.columns)
        self.assertIn("target", df.columns)

    # def test_transactions_y(self):
    #     actual = self.dataset.transactions_x.shape[0] + self.dataset.transactions_y.shape[0]
    #     expected = self.dataset.transactions.shape[0]
    #     self.assertEqual(expected, actual)
    #
    #     expected = (1, 5)
    #     actual = self.dataset.transactions_y.shape
    #     self.assertEqual(expected, actual)
    #
    #     expected = (119, 5)
    #     actual = self.dataset.transactions_x.shape
    #     self.assertEqual(expected, actual)

    def test_transactions_x_y_r(self):
        actual = self.x_y_r.transactions.shape[0]
        expected = self.dataset.transactions.shape[0]
        self.assertEqual(expected, actual)

        expected = (1, 5)
        actual = self.x_y_r.transactions_y.shape
        self.assertEqual(expected, actual)

        expected = (119, 5)
        actual = self.x_y_r.transactions_x.shape
        self.assertEqual(expected, actual)

        expected = (1, 2)
        relevant = self.x_y_r.relevant_set
        actual = relevant.shape
        self.assertEqual(expected, actual)

        a_customer = '00000dbacae5abe5e23885899a1fa44253a17956c6d1c3d25f88aa139fdfc657'
        relevant_slow = datasets.TargetSlow(self.dataset.transactions).relevant_set
        expected = relevant_slow.loc[relevant_slow.customer_id == a_customer, 'target']
        actual = relevant.loc[relevant.customer_id == a_customer, 'target']
        self.assertEqual(expected.values, actual.values)


