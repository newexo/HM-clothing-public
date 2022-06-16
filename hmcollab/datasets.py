import os
import pandas as pd

from . import directories
from hmcollab import transactions


class HMDatasetDirectoryTree:
    def __init__(self, base=None):
        if base is None:
            base = directories.data()
        self._base = base

    def path(self, filename=None):
        return directories.qualifyname(self._base, filename)

    def images(self, filename=None):
        return directories.qualifyname(self.path("images"), filename)

    @property
    def customers(self):
        return self.path("customers.csv")

    @property
    def articles(self):
        return self.path("articles.csv")

    @property
    def transactions(self):
        return self.path("transactions_train.csv")

    @property
    def toy(self):
        return self.path("transactions_toy.csv")

    def image(self, number):
        number = str(number)
        filename = "{}.jpg".format(number)
        prefix = number[:3]
        dir = self.images(prefix)
        return os.path.join(dir, filename)


class HMDataset:
    def __init__(self, tree=None, toy=False):
        if tree is None:
            tree = HMDatasetDirectoryTree()
        self.tree = tree
        self.articles = pd.read_csv(
            self.tree.articles,
            dtype={
                "article_id": object,
                "product_code": object,
                "colour_group_code": object,
            },
        )
        self.customers = pd.read_csv(self.tree.customers)
        if toy:
            self.transactions = pd.read_csv(
                self.tree.toy,
                dtype={
                    "article_id": object,
                },
            )
        else:
            self.transactions = pd.read_csv(
                self.tree.transactions,
                dtype={
                    "article_id": object,
                },
            )
        train, test = transactions.transactions_train_test(
            self.transactions, ids_fraction=0.2
        )
        self.train_x, self.train_y = transactions.split_by_time(train, days=7)
        self.test_x, self.test_y = transactions.split_by_time(test, days=7)
