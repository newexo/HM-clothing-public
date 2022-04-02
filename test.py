#!/usr/bin/env python
import unittest

from hmcollab.tests.test_articles import TestArticles
from hmcollab.tests.test_datasets import TestDatasets
from hmcollab.tests.test_directories import TestDirectories
from hmcollab.tests.test_example import TestExample
from hmcollab.tests.test_transactions import TestTransactions


class CountSuite(object):
    def __init__(self):
        self.count = 0
        self.s = unittest.TestSuite()

    def add(self, tests):
        self.count += 1
        print("%d: %s" % (self.count, tests.__name__))
        self.s.addTest(unittest.makeSuite(tests))


def suite():
    s = CountSuite()

    s.add(TestArticles)
    s.add(TestDatasets)
    s.add(TestDirectories)
    s.add(TestExample)
    s.add(TestTransactions)

    return s.s


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
