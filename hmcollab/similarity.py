import numpy as np

from abc import ABCMeta, abstractmethod


class Similarity(metaclass=ABCMeta):
    @abstractmethod
    def similarity(self, id0, id1):
        pass

    def compare_one(self, predicted, target) -> np.array:
        def is_in_target(i):
            for j in target:
                if self.similarity(i, j):
                    return True
            return False
        comparisons = np.array([is_in_target(i) for i in predicted])
        return comparisons


class IdenticalSimilarity(Similarity):
    def similarity(self, id0, id1):
        return id0 == id1

    # def compare_one(self, predicted, target):
    #     return np.isin(predicted, target)


class DepartmentSimilarity(Similarity):
    def __init__(self, df):
        self.df = df

    def row_from_article_id(self, id0):
        df = self.df[self.df.article_id == id0]
        if df.shape[0]:
            return df.iloc[0]

    def similarity(self, id0, id1):
        row0 = self.row_from_article_id(id0)
        if row0 is None:
            return False
        row1 = self.row_from_article_id(id1)
        if row1 is None:
            return False
        return self.similarity_by_row(row0, row1)

    def similarity_by_iloc(self, index0, index1):
        row0 = self.df.iloc[index0]
        row1 = self.df.iloc[index1]
        return self.similarity_by_row(row0, row1)

    def similarity_by_row(self, row0, row1):
        return row0.department_no == row1.department_no
