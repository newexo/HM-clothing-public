import unittest
from hmcollab.tests import fake_data


class TestFakeData(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_fake_customer_id(self):
        self.assertEqual("00", fake_data.customer_id(0))
        self.assertEqual("05", fake_data.customer_id(5))
        self.assertEqual("0f", fake_data.customer_id(15))

    def test_fake_customers(self):
        n = 16
        df = fake_data.fake_customers(n)
        self.assertEqual(n, df.shape[0])
        self.assertEqual(2, df.shape[1])
        self.assertEqual("customer_id", df.columns[0])
        self.assertEqual("age", df.columns[1])
        self.assertEqual("00", df.iloc[0].customer_id)
        self.assertEqual("0f", df.iloc[15].customer_id)
        self.assertEqual(51, df.iloc[0].age)
        self.assertEqual(87, df.iloc[10].age)
        self.assertEqual(52, df.iloc[15].age)

    def test_articles_random_df(self):
        n = 10
        df = fake_data.articles_random_df(n)
        self.assertEqual(n, df.shape[0])
        self.assertEqual(3, df.shape[1])
        self.assertEqual("article_id", df.columns[0])
        self.assertEqual("color", df.columns[1])
        self.assertEqual("article", df.columns[2])
        self.assertEqual("00", df.iloc[0].article_id)
        self.assertEqual("09", df.iloc[9].article_id)
        self.assertEqual("black", df.iloc[0].color)
        self.assertEqual("white", df.iloc[5].color)
        self.assertEqual("white", df.iloc[9].color)
        self.assertEqual("shoes", df.iloc[0].article)
        self.assertEqual("pants", df.iloc[9].article)

    def test_dates_random(self):
        n = 2
        dates = fake_data.dates_random(n)
        self.assertEqual(8, len(dates))
        self.assertEqual("2021-03-07", dates[0])
        self.assertEqual("2022-04-21", dates[-1])
