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
        self.simple_munger = articles.ArticleFeaturesSimpleFeatures(
            self.dataset.articles
        )
        self.recomendations_by_group = math.ceil(
            self.total_recommendations / self.groups
        )
        self.model = articles.ArticleKNN(
            self.simple_munger, k=self.recomendations_by_group
        )

    def recommend(self, customer):
        recomendation_ids = []
        customer_dummies = self.t.customer_dummies(customer, self.full_article_dummies)
        all_groups = transactions.kmeans_consumer(customer_dummies, k=self.groups)
        for i in range(self.groups):
            one_group = all_groups.cluster_centers_[i]
            _, indices = self.model.nearest(row=one_group)
            for r in indices[0][: self.recomendations_by_group]:
                article_id = self.simple_munger.id_from_index(r)
                recomendation_ids.append(article_id)
        return recomendation_ids

    def recommend_all(self, customer_list):
        df = pd.DataFrame(columns=["prediction"], index=customer_list)
        for c in customer_list:
            recommendations = self.recommend(c)
            df.loc[c] = {"prediction": " ".join(recommendations)}
        df = df.reset_index().rename(columns={"index": "customer_id"})
        return df
