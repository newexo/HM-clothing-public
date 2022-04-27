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

    def image(self, number):
        number = str(number)
        filename = "{}.jpg".format(number)
        prefix = number[:3]
        dir = self.images(prefix)
        return os.path.join(dir, filename)


class HMDataset:
    def __init__(self, tree=None):
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
        self.transactions = pd.read_csv(
            self.tree.transactions,
            dtype={
                "article_id": object,
            },
        )
        self.transactions_x, self.transactions_y = transactions.split_by_time(self.transactions, days=7)
