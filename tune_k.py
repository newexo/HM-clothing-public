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

class ThreeSetsSetup:
    def __init__(self, dataset, features, use_toy=True, threshold=10):
        self.data = dataset
        self.dummies = features.x
        # Only keep customers at train_x and train_y
        train_x_customer_ids_set = set(self.data.train_x.customer_id)
        self.customers_at_y = train_x_customer_ids_set.intersection(
            set(self.data.train_y.customer_id)
        )
        # Only keep customers at val_x and val_y
        val_x_customer_ids_set = set(self.data.val_x.customer_id)
        self.customers_at_vy = val_x_customer_ids_set.intersection(
            set(self.data.val_y.customer_id)
        )
         # Only keep customers at test_x and test_y
        test_x_customer_ids_set = set(self.data.test_x.customer_id)
        self.customers_at_ty = test_x_customer_ids_set.intersection(
            set(self.data.test_y.customer_id)
        ) 
        self.threshold = threshold
        # self.threshold = 300
        # if use_toy:
        #     self.threshold = 10
        self.rel_y = datasets.target_to_relevant(toy.train_y)   # toy=316
        self.rel_vy = datasets.target_to_relevant(toy.val_y)    # toy=107
        self.rel_ty = datasets.target_to_relevant(toy.test_y)   # toy=127

    def try_multiple_k(self, k_list):
        def experiment(k, threshold):
            print('PROCESSING TRAINING SET...')
            model = models.KnnRecommender_for3(
                self.data, self.dummies, groups=k, threshold=threshold, split='train'
            )
            recommendations = model.recommend_all(list(self.customers_at_y))
            t = scoring.relevant(recommendations, self.rel_y)
            score_y = scoring.map_at_k(t)
            print('PROCESSING VALIDATION SET...')
            model = models.KnnRecommender_for3(
                self.data, self.dummies, groups=k, threshold=threshold, split='val'
            )
            recommendations = model.recommend_all(list(self.customers_at_vy))
            t = scoring.relevant(recommendations, self.rel_vy)
            score_vy = scoring.map_at_k(t)
            print('PROCESSING TEST SET...')
            model = models.KnnRecommender_for3(
                self.data, self.dummies, groups=k, threshold=threshold, split='test'
            )
            recommendations = model.recommend_all(list(self.customers_at_ty))
            t = scoring.relevant(recommendations, self.rel_ty)
            score_ty = scoring.map_at_k(t)
            return score_y, score_vy, score_ty
        
        scores_train = []
        scores_validation = []
        scores_test = []
        print('\nKnnRecommender for ThreeSetsSetup with threshold: {}'.format(self.threshold))
        for k in k_list:
            score_y, score_vy, score_ty = experiment(k=k, threshold=self.threshold)
            scores_train.append(score_y)
            scores_validation.append(score_vy)
            scores_test.append(score_ty)
        scores_train_float = [float(x) for x in scores_train]
        scores_validation_float = [float(x) for x in scores_validation]
        scores_test_float = [float(x) for x in scores_test]
        return {
            "k": k_list,
            "map_at_k_train": scores_train_float,
            "map_at_k_validation": scores_validation_float,
            "map_at_k_test": scores_test_float,
        }

if __name__ == "__main__":
    """
    Run KnnRecommender for different k using a yaml file to read the parameters for the experiment.
    If no yaml_path is provided, it will use knn_exp1.yml
    Usage: python3 ./hmcollab/tune_k.py yaml_path

    Note: This setting is trying different sets of experiments (such as different sets of features)
    and trying the different k's on all of them

    """

    if len(sys.argv) > 1:
        yaml_path = sys.argv[1]
    else:
        # yaml_path = directories.experiments("knn_exp1.yml")
        yaml_path = directories.experiments("knn_exp3.yml")

    with open(yaml_path, "r") as file:
        config = yaml.safe_load(file)

    toy = datasets.HMDataset(toy=config["toy"], folds=config["split_strategy"])

    i = 0

    for exp in config["experiments"]:
        i += 1
        the_features = articles.ArticleFeatureMungerSpecificFeatures(
            toy.articles, features=exp["features"], use_article_id=True
        )
        if config["split_strategy"] == 'threesets':
            toy_k = ThreeSetsSetup(toy, features=the_features, threshold=config["threshold"]) 
        else:   
            toy_k = StandardSetup(toy, features=the_features)
        results = toy_k.try_multiple_k(config["k"])

        with open(yaml_path, "a") as outfile:
            yaml.dump({"Results" + "_experiment_" + str(i): results}, outfile, default_flow_style=False)
