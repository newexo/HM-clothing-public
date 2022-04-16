class IdenticalSimilarity:
    def similarity(self, id0, id1):
        return id0 == id1


class DepartmentSimilarity:
    def __init__(self, df):
        self.df = df

    def row_from_article_id(self, id0):
        return self.df[self.df.article_id == id0].iloc[0]

    def similarity(self, id0, id1):
        row0 = self.row_from_article_id(id0)
        row1 = self.row_from_article_id(id1)
        return self.similarity_by_row(row0, row1)

    def similarity_by_iloc(self, index0, index1):
        row0 = self.df.iloc[index0]
        row1 = self.df.iloc[index1]
        return self.similarity_by_row(row0, row1)

    def similarity_by_row(self, row0, row1):
        return row0.department_no == row1.department_no
