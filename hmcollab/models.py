import math
import pandas as pd

from hmcollab import articles
from hmcollab import transactions


class PopularRecommender:
    def __init__(
        self, dataset, full_article_dummies, total_recommendations=12
    ):
        self.dataset = dataset
        self.full_article_dummies = full_article_dummies
        self.total_recommendations = total_recommendations
        self.transactions = self.dataset.train_x
        self.simple_munger = articles.ArticleFeaturesSimpleFeatures(
            self.dataset.articles
        )

    def recommend(self):
        """Return a list of article_ids with the most transactions.
        It will provide the same recommendation for all customers (old and new)"""
        top_df = self.transactions.groupby(['article_id'], as_index=False).size()
        top_df.sort_values(by=['size'], ascending=False, inplace=True)
        return top_df.article_id[:self.total_recommendations].values.tolist()


    def recommend_all(self, customer_list):
        df = pd.DataFrame(columns=["prediction"], index=customer_list)
        df['prediction'] = " ".join(self.recommend())
        df = df.reset_index().rename(columns={"index": "customer_id"})
        return df


class KnnRecommender:
    def __init__(
        self, dataset, full_article_dummies, groups=6, total_recommendations=12
    ):
        self.dataset = dataset
        self.full_article_dummies = full_article_dummies
        self.groups = groups
        self.total_recommendations = total_recommendations
        self.t = transactions.TransactionsByCustomer(self.dataset.train_x)
        # self.t = transactions.TransactionsByCustomer(self.dataset.transactions_x)
        self.simple_munger = articles.ArticleFeaturesSimpleFeatures(
            self.dataset.articles
        )
        self.recomendations_by_group = math.ceil(
            self.total_recommendations / self.groups
        )
        self.model = articles.ArticleKNN(
            self.simple_munger, k=self.recomendations_by_group
        )

    def recommend(self, customer, drop_duplicates=True):
        recomendation_ids = []
        customer_dummies = self.t.customer_dummies(customer, self.full_article_dummies)
        if drop_duplicates:
            customer_dummies = customer_dummies.drop_duplicates()
        min_k = self.groups
        if customer_dummies.shape[0] < self.groups:
            min_k = customer_dummies.shape[0]   # so far it will produce less recommendations for this customer
        all_groups = transactions.kmeans_consumer(customer_dummies, k=min_k)
        for i in range(min_k):
            one_group = all_groups.cluster_centers_[i]
            _, indices = self.model.nearest(row=one_group)
            for r in indices[0][: self.recomendations_by_group]:
                article_id = self.simple_munger.id_from_index(r)
                recomendation_ids.append(article_id)
        return recomendation_ids

    def recommend_all(self, customer_list, drop_duplicates=True):
        df = pd.DataFrame(columns=["prediction"], index=customer_list)
        for c in customer_list:
            recommendations = self.recommend(c, drop_duplicates=drop_duplicates)
            df.loc[c] = {"prediction": " ".join(recommendations)}
        df = df.reset_index().rename(columns={"index": "customer_id"})
        return df


def filter_articles(trans_df, threshold=50):
    # Compute the number of transactions by article_id. A good threshold for toy is 50, and 300 for full dataset
    # returns:
    #    filtered_article_id: list
    freqs_df = trans_df.groupby(by=['article_id'], as_index=False).size()
    freqs_df.reset_index(inplace=True)
    freqs_df.drop(columns=['index'], inplace=True)
    freqs_df.rename(columns={'size':'transactions'}, inplace=True)
    filtered_article_id = freqs_df.loc[freqs_df.transactions>threshold, 'article_id'].to_list()
    return filtered_article_id


class KnnRecommender_new:
    def __init__(
        self, dataset, full_article_dummies, groups=6, total_recommendations=12, threshold=50
    ):
        self.dataset = dataset
        self.full_article_dummies = full_article_dummies
        self.groups = groups
        self.total_recommendations = total_recommendations
        self.t = transactions.TransactionsByCustomer_new(self.dataset.train_x)
        # self.t = transactions.TransactionsByCustomer(self.dataset.transactions_x)
        filtered_art_ids = filter_articles(self.dataset.train_x, threshold=threshold)
        self.filtered_dummies = full_article_dummies[full_article_dummies.article_id.isin(
            filtered_art_ids)].drop(columns="article_id")
        filtered_articles = self.dataset.articles.loc[
            self.dataset.articles.article_id.isin(filtered_art_ids), :]
        self.filtered_articles_main_features = articles.ArticleFeaturesSimpleFeatures(
            filtered_articles
        )     # This should be the filtered one
        self.recomendations_by_group = math.ceil(
            self.total_recommendations / self.groups
        )
        self.model = articles.ArticleKNN_new(
            self.filtered_dummies, k=self.recomendations_by_group
        )


    def recommend(self, customer, drop_duplicates=True):
        recomendation_ids = []
        # We need full dummies for kmeans
        customer_dummies = self.t.customer_dummies(customer, self.full_article_dummies)
        if drop_duplicates:
            customer_dummies = customer_dummies.drop_duplicates()     # no article_id
        min_k = self.groups
        if customer_dummies.shape[0] < self.groups:
            min_k = customer_dummies.shape[0]   # so far it will produce less recommendations for this customer
        all_groups = transactions.kmeans_consumer(customer_dummies, k=min_k)
        for i in range(min_k):
            one_group = all_groups.cluster_centers_[i]
            # We can use filtered dummies here
            _, indices = self.model.nearest(row=one_group)    # same shape as filtered (with article_id)
            for r in indices[0][: self.recomendations_by_group]:
                article_id = self.filtered_articles_main_features.id_from_index(r)
                recomendation_ids.append(article_id)
        return recomendation_ids

    def recommend_all(self, customer_list, drop_duplicates=True):
        df = pd.DataFrame(columns=["prediction"], index=customer_list)
        for c in customer_list:
            recommendations = self.recommend(c, drop_duplicates=drop_duplicates)
            df.loc[c] = {"prediction": " ".join(recommendations)}
        df = df.reset_index().rename(columns={"index": "customer_id"})
        return df