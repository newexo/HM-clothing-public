from abc import ABCMeta, abstractmethod

import pandas as pd


class ArticleMunger(metaclass=ABCMeta):
    def __init__(self, df: pd.DataFrame, use_article_id=False):
        self.df = df
        self.use_article_id = use_article_id
        self.x = self._to_array()

    @abstractmethod
    def _to_array(self):
        pass

    def id_from_index(self, i):
        return self.df.iloc[i].article_id


class ArticleFeatureMunger(ArticleMunger, metaclass=ABCMeta):
    @abstractmethod
    def features(self):
        pass

    def _to_array(self):
        features = self.features()
        if self.use_article_id:
            return pd.get_dummies(
                self.df[["article_id"] + features], columns=features, prefix=features
            )
        return pd.get_dummies(self.df[features], columns=features, prefix=features)


class ArticleFeaturesSimpleFeatures(ArticleFeatureMunger):
    def features(self):
        return [
            "product_type_no",
            "product_group_name",
            "graphical_appearance_no",
            "colour_group_code",
            "perceived_colour_value_id",
            "perceived_colour_master_id",
            "department_no",
            "index_code",
            "index_group_no",
            "section_no",
            "garment_group_no",
        ]





def filter_articles_a(trans_df, threshold=50):
    # Compute the number of transactions by article_id. A good threshold for toy is 50, and 300 for full dataset
    # returns:
    #    filtered_article_id: list
    freqs_df = trans_df.groupby(by=['article_id'], as_index=False).size()
    freqs_df.reset_index(inplace=True)
    freqs_df.drop(columns=['index'], inplace=True)
    freqs_df.rename(columns={'size':'transactions'}, inplace=True)
    filtered_article_id = freqs_df.loc[freqs_df.transactions>threshold, 'article_id'].to_list()
    return filtered_article_id


class ArticleKNN_new:
    def __init__(self, dummies, k=20):
        self.k = k
        # self.a = articles
        self.model = NearestNeighbors(n_neighbors=k).fit(dummies.values)

    def nearest(self, index=None, id=None, row=None):
        # if id is not None:
        #     matches = self.a.df[self.a.df.article_id == id].index
        #     index = matches[0]
        #
        # if index is not None:
        #     row = self.a.x.values[index]

        row = row.reshape((1, -1))

        return self.model.kneighbors(row)