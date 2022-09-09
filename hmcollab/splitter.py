import datetime

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from hmcollab.three_part_dataset import ThreePartDataset


def split_by_time(df, days):
    """Split transactions dataframe by a cutoff number of days from
    the last transaction at column t_dat"""
    df["t_dat"] = pd.to_datetime(df["t_dat"].copy(), format="%Y-%m-%d")
    last = df["t_dat"].max()
    cutoff_date = last - datetime.timedelta(days=days)
    older = df[df.t_dat < cutoff_date]
    newer = df[df.t_dat >= cutoff_date]
    return older, newer


def split_ids(a_set, fraction, random_state=42):
    full_ids = a_set.customer_id.unique()
    ids_train, ids_test = train_test_split(
        full_ids, test_size=fraction, random_state=random_state
    )
    return ids_train, ids_test


def transactions_train_test(a_set, ids_tr, ids_te):
    # a_set: transactions dataset
    train = a_set.loc[a_set.customer_id.isin(ids_tr), :]
    test = a_set.loc[a_set.customer_id.isin(ids_te), :]
    return train, test


class CustomerPortion:
    def __init__(self, customers_ids):
        self.customers_ids = customers_ids

    def split(self, dataset: ThreePartDataset):
        customers = dataset.customers.loc[
            dataset.customers.customer_id.isin(self.customers_ids), :
        ]
        transactions = dataset.transactions.loc[
            dataset.transactions.customer_id.isin(self.customers_ids), :
        ]
        article_ids = transactions.article_id.unique()
        articles = dataset.articles.loc[
            dataset.articles.article_id.isin(article_ids), :
        ]
        return ThreePartDataset(articles, customers, transactions)


class SplitByCustomerStrategy:
    def __init__(self, dataset: ThreePartDataset, customer_id_partition):
        partition = []
        for ids in customer_id_partition:
            cp = CustomerPortion(ids)
            partition.append(cp.split(dataset))
        self.partition = partition

    @staticmethod
    def _get_random_state(random_state):
        if type(random_state) is int:
            random_state = np.random.RandomState(random_state)
        return random_state


class SplitByCustomerTrainTestStrategy(SplitByCustomerStrategy):
    def __init__(self, dataset: ThreePartDataset, fraction, random_state=42):
        random_state = SplitByCustomerStrategy._get_random_state(random_state)
        ids_train, ids_test = split_ids(
            dataset.customers, fraction, random_state=random_state
        )
        super().__init__(dataset, [ids_train, ids_test])

    @property
    def train(self):
        return self.partition[0]

    @property
    def test(self):
        return self.partition[1]


class SplitByCustomerTrainTestValidationStrategy(SplitByCustomerStrategy):
    def __init__(
        self,
        dataset: ThreePartDataset,
        test_fraction,
        validation_fraction,
        random_state=42,
    ):
        random_state = SplitByCustomerStrategy._get_random_state(random_state)
        full_ids = dataset.customers.customer_id.unique()
        partial_ids, ids_test = train_test_split(
            full_ids, test_size=test_fraction, random_state=random_state
        )
        ids_train, ids_validation = train_test_split(
            partial_ids,
            test_size=validation_fraction / (1.0 - test_fraction),
            random_state=random_state,
        )
        super().__init__(dataset, [ids_train, ids_test, ids_validation])

    @property
    def train(self):
        return self.partition[0]

    @property
    def test(self):
        return self.partition[1]

    @property
    def validation(self):
        return self.partition[2]
