import unittest
import os

from hmcollab import datasets
from hmcollab import directories


class TestDatasets(unittest.TestCase):
    def setUp(self):
        self.tree = datasets.HMDatasetDirectoryTree(base=directories.testdata())
        self.dataset = datasets.HMDataset(tree=self.tree)

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
        self.assertEqual(17, self.dataset.articles.shape[0])

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
        self.assertEqual(2, self.dataset.customers.shape[0])

    def test_transactions(self):
        expected = ['t_dat', 'customer_id', 'article_id', 'price', 'sales_channel_id']
        actual = list(self.dataset.transactions.columns)
        self.assertEqual(expected, actual)
        self.assertEqual(3, self.dataset.transactions.shape[0])
