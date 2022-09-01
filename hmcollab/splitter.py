import datetime

import pandas as pd
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


def split_ids(a_set, fraction):
    full_ids = a_set.customer_id.unique()
    ids_train, ids_test = train_test_split(
        full_ids, test_size=fraction, random_state=42
    )
    return ids_train, ids_test


def transactions_train_test(a_set, ids_tr, ids_te):
    # a_set: transactions dataset
    train = a_set.loc[a_set.customer_id.isin(ids_tr), :]
    test = a_set.loc[a_set.customer_id.isin(ids_te), :]
    return train, test


class CustomerPortion:
    def __init__(self, ids):
        self.ids = ids

    def split(self, dataset: ThreePartDataset):
        a_set = dataset.transactions
        transactions = a_set.loc[a_set.customer_id.isin(self.ids), :]
        return ThreePartDataset(dataset.articles, dataset.customers, transactions)
