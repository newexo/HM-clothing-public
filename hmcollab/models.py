import math
import pandas as pd

from hmcollab import articles
from hmcollab import transactions


class KnnRecommender:
    def __init__(
        self, dataset, full_article_dummies, groups=6, total_recommendations=12
    ):
        self.dataset = dataset
        self.full_article_dummies = full_article_dummies
        self.groups = groups
        self.total_recommendations = total_recommendations
        self.t = transactions.TransactionsByCustomer(self.dataset.transactions)
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


def recommender_by_customer(
    customer, dataset, full_article_dummies, groups=6, total_recommendations=12
):
    """
    customer:
        The customer id
    groups:
        The number of groups we want to cluster the customer transactions
    """
    recomendations_by_group = math.ceil(total_recommendations / groups)
    recomendation_ids = []
    t = transactions.TransactionsByCustomer(dataset.transactions)
    customer_dummies = t.customer_dummies(customer, full_article_dummies)
    all_groups = transactions.kmeans_consumer(customer_dummies, k=groups)
    simple_munger = articles.ArticleFeaturesSimpleFeatures(dataset.articles)
    model = articles.ArticleKNN(simple_munger, k=recomendations_by_group)
    for i in range(groups):
        one_group = all_groups.cluster_centers_[i]
        _, indices = model.nearest(row=one_group)
        for r in indices[0][:recomendations_by_group]:
            article_id = simple_munger.id_from_index(r)
            recomendation_ids.append(article_id)
    return recomendation_ids


def recommender(customers, dataset, full_dummies):
    """
    customers: list
        customer_id from the group of customer for which the recommendation is needed
    """
    df = pd.DataFrame(columns=["prediction"], index=customers)
    for c in customers:
        recommendations = recommender_by_customer(c, dataset, full_dummies)
        df.loc[c] = {"prediction": " ".join(recommendations)}
    df = df.reset_index().rename(columns={"index": "customer_id"})
    return df
