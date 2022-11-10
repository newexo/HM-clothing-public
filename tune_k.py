from hmcollab import datasets
from hmcollab import articles
from hmcollab import models
from hmcollab import scoring
from hmcollab import directories

import yaml
import sys


class StandardSetup:
    def __init__(self, dataset, features, use_toy=True):
        self.data = dataset
        self.dummies = features.x
        # Only keep customers at train_x and train_y
        train_x_customer_ids_set = set(self.data.train_x.customer_id)
        self.customers_at_y = train_x_customer_ids_set.intersection(
            set(self.data.train_y.customer_id)
        )  # toy=539
        self.customers_at_vy = train_x_customer_ids_set.intersection(
            set(self.data.train_vy.customer_id)
        )  # toy=607
        self.threshold = 300
        if use_toy:
            self.threshold = 50
        self.rel_vy = datasets.target_to_relevant(toy.train_vy)

    def try_multiple_k(self, k_list):
        scores_validation = []
        scores_test = []
        for k in k_list:
            model = models.KnnRecommender(
                self.data, self.dummies, groups=k, threshold=self.threshold
            )
            recommendations = model.recommend_all(list(self.customers_at_vy))
            t = scoring.relevant(recommendations, self.rel_vy)
            score_vy = scoring.map_at_k(t)
            scores_validation.append(score_vy)
            recommendations = model.recommend_all(list(self.customers_at_y))
            t = scoring.relevant(recommendations, self.data.relevant_set)
            score_y = scoring.map_at_k(t)
            scores_test.append(score_y)
        scores_validation_float = [float(x) for x in scores_validation]
        scores_test_float = [float(x) for x in scores_test]
        return {
            "k": k_list,
            "map_at_k_validation": scores_validation_float,
            "map_at_k_test": scores_test_float,
        }


if __name__ == "__main__":
    """
    Run KnnRecommender for different k using a yaml file to read the parameters for the experiment.
    If no yaml_path is provided, it will use knn_exp1.yml
    Usage: python3 ./hmcollab/tune_k.py yaml_path

    Fields needed at the yaml file:
    k: [2,4, 5, 6, 8]   # a list of integers
    toy : True    # True to use toy dataset
    threshold: 50    # Threshold to filter articles. Suggested: 50 for toy, 300 for full
    split_strategy: "standard"    # Splitting strategy. So far only "standard"
    """

    if len(sys.argv) > 1:
        yaml_path = sys.argv[1]
    else:
        yaml_path = directories.experiments("knn_exp1.yml")

    with open(yaml_path, "r") as file:
        config = yaml.safe_load(file)

    toy = datasets.HMDataset(toy=config["toy"], folds=config["split_strategy"])
    features1 = articles.ArticleFeaturesSimpleFeatures(
        toy.articles, use_article_id=True
    )
    toy_k = StandardSetup(toy, features=features1)
    results = toy_k.try_multiple_k(config["k"])

    with open(yaml_path, "a") as outfile:
        yaml.dump({"Results": results}, outfile, default_flow_style=False)
