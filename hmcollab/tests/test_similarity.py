import unittest

from hmcollab import datasets
from hmcollab import directories
from hmcollab.directory_tree import HMDatasetDirectoryTree
from hmcollab.similarity import (
    IdenticalSimilarity,
    DepartmentSimilarityByColumn,
    ProductCodeSimilarityByColumn,
    ColourGroupCodeSimilarityByColumn,
    GarmentGroupNoSimilarityByColumn,
    get_similarity,
)


# TODO: replace these tests with unit tests
class TestSimilarity(unittest.TestCase):
    def setUp(self):
        self.tree = HMDatasetDirectoryTree(base=directories.testdata())
        self.dataset = datasets.HMDataset(tree=self.tree)
        self.id0 = "0110065001"
        self.id1 = "0111565001"
        self.id2 = "0111586001"
        self.id3 = "0111586001"
        self.ids = [self.id0, self.id1, self.id2, self.id3]

    def tearDown(self):
        pass

    def get_department_similarity(self):
        return DepartmentSimilarityByColumn(self.dataset.articles)

    def test_identity_similarity(self):
        sim = IdenticalSimilarity()
        self.assertFalse(sim.similarity(1, 2))
        self.assertTrue(sim.similarity(2, 2))

    def test_row_from_article_id(self):
        sim = self.get_department_similarity()
        row = sim.row_from_article_id(self.id3)
        self.assertEqual(self.id3, row.article_id)
        self.assertEqual("0111586", row.product_code)

        self.assertIsNone(sim.row_from_article_id("not an id"))

    def test_department_similarity_by_article_id(self):
        sim = self.get_department_similarity()

        # the second and third rows have same department_no, but the first row does not
        self.assertTrue(sim.similarity(self.id1, self.id2))
        self.assertFalse(sim.similarity(self.id0, self.id1))
        expected = [False, True, True, True]
        actual = [sim.similarity(self.id1, i) for i in self.ids]
        self.assertEqual(expected, actual)

    def test_department_similarity_by_article_id_missing_id(self):
        sim = self.get_department_similarity()

        not_an_id = "not an id"
        self.assertFalse(sim.similarity(self.id1, not_an_id))
        self.assertFalse(sim.similarity(not_an_id, self.id1))

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

    def test_identical_compare_one_general(self):
        sim = IdenticalSimilarity()
        p = [
            "0351484002",
            "0723529001",
            "0811835004",
            "0689898002",
            "0640174001",
            "0797065001",
            "0599580055",
            "0811927004",
            "0811925005",
            "0800436010",
            "0666448006",
            "0663713001",
        ]
        r = ["0794321007"]
        expected = [False] * 12
        actual = list(sim.compare_one(p, r))
        self.assertEqual(expected, actual)

        r = ["0351484002"]
        expected = [True] + [False] * 11
        actual = list(sim.compare_one(p, r))
        self.assertEqual(expected, actual)

        r = ["0794321007", "0351484002", "0689898002", "0800436010"]
        actual = list(sim.compare_one(p, r))
        expected = [
            True,
            False,
            False,
            True,
            False,
            False,
            False,
            False,
            False,
            True,
            False,
            False,
        ]
        self.assertEqual(expected, actual)

    def test_identical_compare_one(self):
        sim = IdenticalSimilarity()
        p = self.ids

        r = []
        expected = [False] * 4
        actual = list(sim.compare_one(p, r))
        self.assertEqual(expected, actual)

        r = ["0794321007"]
        expected = [False] * 4
        actual = list(sim.compare_one(p, r))
        self.assertEqual(expected, actual)

        r = [self.id2]
        expected = [False, False, True, True]
        actual = list(sim.compare_one(p, r))
        self.assertEqual(expected, actual)

        r = [self.id1]
        actual = list(sim.compare_one(p, r))
        expected = [False, True, False, False]
        self.assertEqual(expected, actual)

        r = [self.id1, self.id0]
        actual = list(sim.compare_one(p, r))
        expected = [True, True, False, False]
        self.assertEqual(expected, actual)

    def test_department_compare_one(self):
        sim = self.get_department_similarity()
        p = self.ids

        r = []
        expected = [False] * 4
        actual = list(sim.compare_one(p, r))
        self.assertEqual(expected, actual)

        r = ["0794321007"]
        expected = [False] * 4
        actual = list(sim.compare_one(p, r))
        self.assertEqual(expected, actual)

        r = [self.id2]
        expected = [False, True, True, True]
        actual = list(sim.compare_one(p, r))
        self.assertEqual(expected, actual)

        r = [self.id1]
        actual = list(sim.compare_one(p, r))
        expected = [False, True, True, True]
        self.assertEqual(expected, actual)

        r = [self.id1, self.id0]
        actual = list(sim.compare_one(p, r))
        expected = [True, True, True, True]
        self.assertEqual(expected, actual)

    def test_product_code_similarity(self):
        sim = ProductCodeSimilarityByColumn(self.dataset.articles)
        i = "0118458038"
        j = "0118458039"
        row_i = sim.row_from_article_id(i)
        row_j = sim.row_from_article_id(j)
        self.assertEqual(row_i.product_code, row_j.product_code)
        self.assertTrue(sim.similarity(i, j))

    def test_colour_group_code_similarity(self):
        sim = ColourGroupCodeSimilarityByColumn(self.dataset.articles)
        i = "0887593002"
        j = "0611584007"
        row_i = sim.row_from_article_id(i)
        row_j = sim.row_from_article_id(j)
        self.assertEqual(row_i.colour_group_code, row_j.colour_group_code)
        self.assertTrue(sim.similarity(i, j))

    def test_garment_group_no_similarity(self):
        sim = GarmentGroupNoSimilarityByColumn(self.dataset.articles)
        i = "0866383001"
        j = "0599580055"
        row_i = sim.row_from_article_id(i)
        row_j = sim.row_from_article_id(j)
        self.assertEqual(row_i.garment_group_no, row_j.garment_group_no)
        self.assertTrue(sim.similarity(i, j))

    def test_get_similarity(self):
        sim = get_similarity("product_code", None)
        self.assertEqual("product_code", sim.column_name)

        sim = get_similarity("colour_group_code", None)
        self.assertEqual("colour_group_code", sim.column_name)

        sim = get_similarity("department_no", None)
        self.assertEqual("department_no", sim.column_name)

        sim = get_similarity("garment_group_no", None)
        self.assertEqual("garment_group_no", sim.column_name)

        sim = get_similarity("not a similarity", None)
        self.assertIsInstance(sim, IdenticalSimilarity)
