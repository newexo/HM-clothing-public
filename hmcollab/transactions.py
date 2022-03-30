import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


class TransactionsByCustomer:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def all_article_ids(self, customer):
        return self.df.loc[self.df.customer_id == customer, 'article_id']


def kmeans_consumer(customer, transactions_df, full_articles_dummy, k=1):
    basket = TransactionsByCustomer(transactions_df).all_article_ids(customer)
    customer_dummies = full_articles_dummy.merge(basket, left_on='article_id', right_on=basket,
                                                 how='right').drop(columns='article_id')
    scaled_features = StandardScaler().fit_transform(customer_dummies)
    kmeans = KMeans(init="k-means++",
         n_clusters=k,
         n_init=10,
         max_iter=300,
         random_state=42)
    return kmeans.fit(scaled_features)
