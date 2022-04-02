import pandas as pd
from sklearn.cluster import KMeans


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


def kmeans_consumer(customer, transactions_df, full_articles_dummy, k=1):
    # Note: Do not scale dummy features
    basket = TransactionsByCustomer(transactions_df).all_article_ids(customer)
    customer_dummies = full_articles_dummy.merge(
        basket, on="article_id", how="right"
    ).drop(columns="article_id")
    kmeans = KMeans(
        init="k-means++", n_clusters=k, n_init=10, max_iter=300, random_state=42
    )
    return kmeans.fit(customer_dummies)
