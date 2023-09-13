import math

import pandas as pd
from sklearn.neighbors import NearestNeighbors

from hmcollab import transactions


class PopularRecommender:
    def __init__(self, dataset, total_recommendations=12):
        self.dataset = dataset
        self.total_recommendations = total_recommendations
        self.transactions = self.dataset.train_x

    def recommend(self):
        """Return a list of article_ids with the most transactions.
        It will provide the same recommendation for all customers (old and new)"""
        top_df = self.transactions.groupby(["article_id"], as_index=False).size()
        top_df.sort_values(by=["size"], ascending=False, inplace=True)
        return top_df.article_id[: self.total_recommendations].values.tolist()

    def recommend_all(self, customer_list):
        df = pd.DataFrame(columns=["prediction"], index=customer_list)
        df["prediction"] = " ".join(self.recommend())
        df = df.reset_index().rename(columns={"index": "customer_id"})
        return df


class ArticleKNN:
    def __init__(self, dummies, k=20):
        self.k = k
        if "article_id" in dummies.columns:
            dummies = dummies.drop(columns=["article_id"])
        self.model = NearestNeighbors(n_neighbors=k).fit(dummies.values)

    def nearest(self, row=None):
        row = row.reshape((1, -1))

        return self.model.kneighbors(row)


def filter_articles(trans_df, threshold=50):
    # Compute the number of transactions by article_id. A good threshold for toy is 50, and 300 for full dataset
    # returns:
    #    filtered_article_id: list
    freqs_df = trans_df.groupby(by=["article_id"], as_index=False).size()
    freqs_df.reset_index(inplace=True)
    freqs_df.drop(columns=["index"], inplace=True)
    freqs_df.rename(columns={"size": "transactions"}, inplace=True)
    filtered_article_id = freqs_df.loc[
        freqs_df.transactions > threshold, "article_id"
    ].to_list()
    return filtered_article_id


class KnnRecommender:
    def __init__(
        self,
        dataset,
        full_article_dummies,
        groups=6,
        total_recommendations=12,
        threshold=50,
    ):
        self.dataset = dataset
        self.full_article_dummies = full_article_dummies
        self.groups = groups
        self.total_recommendations = total_recommendations
        self.t = transactions.TransactionsByCustomer(self.dataset.train_x)
        filtered_art_ids = filter_articles(self.dataset.train_x, threshold=threshold)
        # I should select features here instead of full
        self.filtered_dummies = full_article_dummies[
            full_article_dummies.article_id.isin(filtered_art_ids)
        ]
        self.recomendations_by_group = math.ceil(
            self.total_recommendations / self.groups
        )
        self.model = ArticleKNN(self.filtered_dummies, k=self.recomendations_by_group)

    def recommend(self, customer, drop_duplicates=True):
        recomendation_ids = []
        # We need full dummies for kmeans
        customer_dummies = self.t.customer_dummies(customer, self.full_article_dummies)
        if customer_dummies.shape[0] == 0:
            print("No dummies for customer: ", customer)
        if drop_duplicates:
            customer_dummies = customer_dummies.drop_duplicates()  # no article_id
        min_k = self.groups
        if customer_dummies.shape[0] < self.groups:
            min_k = customer_dummies.shape[
                0
            ]  # so far it will produce less recommendations for this customer
        all_groups = transactions.kmeans_consumer(customer_dummies, k=min_k)
        for i in range(min_k):
            one_group = all_groups.cluster_centers_[i]
            # We can use filtered dummies here
            _, indices = self.model.nearest(
                row=one_group
            )  # same shape as filtered (with article_id)
            for r in indices[0][: self.recomendations_by_group]:
                article_id = self.filtered_dummies.iloc[r].article_id
                recomendation_ids.append(article_id)
        return recomendation_ids

    def recommend_all(self, customer_list, drop_duplicates=True):
        df = pd.DataFrame(columns=["prediction"], index=customer_list)
        for c in customer_list:
            recommendations = self.recommend(c, drop_duplicates=drop_duplicates)
            df.loc[c] = {"prediction": " ".join(recommendations)}
        df = df.reset_index().rename(columns={"index": "customer_id"})
        return df


class KnnRecommender_for3(KnnRecommender):
    def __init__(
        self,
        dataset,
        full_article_dummies,
        groups=6,
        total_recommendations=12,
        threshold=50,
        split="train",
    ):
        KnnRecommender.__init__(
            self,
            dataset,
            full_article_dummies,
            groups,
            total_recommendations,
            threshold,
        )
        if split == "train":
            self.t = transactions.TransactionsByCustomer(self.dataset.train_x)
            filtered_art_ids = filter_articles(
                self.dataset.train_x, threshold=threshold
            )
            print("Articles at train: ", len(filtered_art_ids))
        if split == "val":
            self.t = transactions.TransactionsByCustomer(self.dataset.val_x)
            filtered_art_ids = filter_articles(self.dataset.val_x, threshold=threshold)
            print("Articles at val: ", len(filtered_art_ids))
        if split == "test":
            self.t = transactions.TransactionsByCustomer(self.dataset.test_x)
            filtered_art_ids = filter_articles(self.dataset.test_x, threshold=threshold)
            print("Articles at test: ", len(filtered_art_ids))
        # I should select features here instead of full
        self.filtered_dummies = full_article_dummies[
            full_article_dummies.article_id.isin(filtered_art_ids)
        ]
        self.recomendations_by_group = math.ceil(
            self.total_recommendations / self.groups
        )
        self.model = ArticleKNN(self.filtered_dummies, k=self.recomendations_by_group)
