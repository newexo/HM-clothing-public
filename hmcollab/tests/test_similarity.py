import unittest

from hmcollab import datasets
from hmcollab import directories
from hmcollab.similarity import IdenticalSimilarity, DepartmentSimilarity


class TestSimilarity(unittest.TestCase):
    def setUp(self):
        self.tree = datasets.HMDatasetDirectoryTree(base=directories.testdata())
        self.dataset = datasets.HMDataset(tree=self.tree)
        self.id0 = "0110065001"
        self.id1 = "0111565001"
        self.id2 = "0111586001"
        self.id3 = "0111586001"

    def tearDown(self):
        pass

    def get_department_similarity(self):
        return DepartmentSimilarity(self.dataset.articles)

    def test_identity_similarity(self):
        sim = IdenticalSimilarity()
        self.assertFalse(sim.similarity(1, 2))
        self.assertTrue(sim.similarity(2, 2))

    def test_row_from_article_id(self):
        sim = self.get_department_similarity()
        row = sim.row_from_article_id(self.id3)
        self.assertEqual(self.id3, row.article_id)
        self.assertEqual("0111586", row.product_code)

    def test_department_similarity_by_article_id(self):
        sim = self.get_department_similarity()

        # the second and third rows have same department_no, but the first row does not
        self.assertTrue(sim.similarity(self.id1, self.id2))
        self.assertFalse(sim.similarity(self.id0, self.id1))

    def test_similarity_by_row(self):
        sim = self.get_department_similarity()

        row0 = self.dataset.articles.iloc[0]
        row1 = self.dataset.articles.iloc[1]
        row2 = self.dataset.articles.iloc[2]

        # the second and third rows have same department_no, but the first row does not
        self.assertTrue(sim.similarity_by_row(row1, row2))
        self.assertFalse(sim.similarity_by_row(row0, row1))

    def test_similarity_by_iloc(self):
        sim = self.get_department_similarity()

        # the second and third rows have same department_no, but the first row does not
        self.assertTrue(sim.similarity_by_iloc(1, 2))
        self.assertFalse(sim.similarity_by_iloc(0, 1))
