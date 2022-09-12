import os
import pandas as pd

import hmcollab.splitter
from . import directories
from .three_part_dataset import ThreePartDataset


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
    def transactions_y_by_customer(self):
        return self.path("target_set_7d_75481u.csv")

    @property
    def transactions_y_by_customer_exists(self):
        return os.path.exists(self.transactions_y_by_customer)

    @property
    def toy(self):
        return self.path("transactions_toy.csv")

    def image(self, number):
        number = str(number)
        filename = "{}.jpg".format(number)
        prefix = number[:3]
        dir = self.images(prefix)
        return os.path.join(dir, filename)


class TargetSlow:
    def __init__(self, transactions_df):
        def create_relevant_set(df, customer_list):
            # df is a transactions DataFrame
            relevant = pd.DataFrame(columns=["target"], index=customer_list)
            for c in customer_list:
                relevant.loc[c] = {
                    "target": " ".join(df.loc[df.customer_id == c, "article_id"])
                }
            relevant = relevant.reset_index().rename(columns={"index": "customer_id"})
            return relevant

        self.transactions = transactions_df
        self.transactions_x, self.transactions_y = hmcollab.splitter.split_by_time(
            self.transactions, days=7
        )
        target_ids = self.transactions_y.customer_id.unique()
        self.relevant_set = create_relevant_set(self.transactions_y, target_ids)


class Target:
    def __init__(self, transactions_df):
        def relevant_dict(tup):
            return " ".join(tup[1].loc[:, "article_id"].tolist())

        self.transactions = transactions_df
        self.transactions_x, self.transactions_y = hmcollab.splitter.split_by_time(
            self.transactions, days=7
        )
        grouped = self.transactions_y.loc[:, ["customer_id", "article_id"]].groupby(
            ["customer_id"]
        )
        by_row = {t[0]: relevant_dict(t) for t in grouped}
        self.relevant_set = pd.DataFrame.from_dict(
            by_row, orient="index", columns=["target"]
        )
        self.relevant_set.reset_index(inplace=True)
        self.relevant_set.rename(columns={"index": "customer_id"}, inplace=True)


class HMDataset(ThreePartDataset):
    def __init__(self, tree=None, toy=False, folds='twosets'):
        if tree is None:
            tree = HMDatasetDirectoryTree()
        self.tree = tree
        articles = pd.read_csv(
            self.tree.articles,
            dtype={
                "article_id": object,
                "product_code": object,
                "colour_group_code": object,
            },
        )
        customers = pd.read_csv(self.tree.customers)

        self.relevant_set = None
        if toy:
            transactions = pd.read_csv(
                self.tree.toy,
                dtype={
                    "article_id": object,
                },
            )
        else:
            transactions = pd.read_csv(
                self.tree.transactions,
                dtype={
                    "article_id": object,
                },
            )
            if self.tree.transactions_y_by_customer_exists:
                self.relevant_set = pd.read_csv(
                    self.tree.transactions_y_by_customer,
                    dtype={
                        "article_id": object,
                    },
                )
                (
                    self.transactions_x,
                    self.transactions_y,
                ) = hmcollab.splitter.split_by_time(transactions, days=7)

        ThreePartDataset.__init__(self, articles, customers, transactions)

        if self.relevant_set is None:
            target = Target(self.transactions)
            self.relevant_set = target.relevant_set
            self.transactions_x, self.transactions_y = (
                target.transactions_x,
                target.transactions_y,
            )

        if folds=='twosets':
            ids_train, ids_test = hmcollab.splitter.split_ids(
                self.transactions, fraction=0.2
            )
            self.train_x, self.test_x = hmcollab.splitter.transactions_train_test(
                self.transactions_x, ids_train, ids_test
            )
            self.train_y, self.test_y = hmcollab.splitter.transactions_train_test(
                self.transactions_y, ids_train, ids_test
            )

        if folds=='threesets':
            # Train: 60%, val: 20%, test: 20%
            # Creating test set
            ids_train, ids_test = hmcollab.splitter.split_ids(
                self.transactions, fraction=0.2
            )
            train_x, self.test_x = hmcollab.splitter.transactions_train_test(
                self.transactions_x, ids_train, ids_test
            )
            train_y, self.test_y = hmcollab.splitter.transactions_train_test(
                self.transactions_y, ids_train, ids_test
            )
            # Creating training and validation sets
            ids_train, ids_val = hmcollab.splitter.split_ids(
                train_x, fraction=0.25
            )  # 25% of training is 20% from the total: 80(.25)=20
            self.train_x, self.val_x = hmcollab.splitter.transactions_train_test(
                train_x, ids_train, ids_val
            )
            self.train_y, self.val_y = hmcollab.splitter.transactions_train_test(
                train_y, ids_train, ids_val
            )

        if folds=='standard':
            # Splitting by leave last week. Note that train_x is used to train all (train, validations and test)
            # train_vy is the target variable for validation to use with train data
            self.train_y = self.transactions_y
            self.train_x, self.train_vy = hmcollab.splitter.split_by_time(self.transactions_x, days=7)

