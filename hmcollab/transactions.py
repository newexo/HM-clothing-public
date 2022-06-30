import datetime
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split


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


class TransactionsByCustomer:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def all_article_ids(self, customer):
        return self.df.loc[self.df.customer_id == customer, "article_id"]

    def customer_dummies(self, customer, full_articles_dummy):
        basket = self.all_article_ids(customer)
        return full_articles_dummy.merge(basket, on="article_id", how="right").drop(
            columns="article_id"
        )


def kmeans_consumer(customer_dummies, k=1):
    # Note: Do not scale dummy features
    kmeans = KMeans(
        init="k-means++", n_clusters=k, n_init=10, max_iter=300, random_state=42
    )
    return kmeans.fit(customer_dummies)
