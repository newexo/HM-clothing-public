#!/usr/bin/env python
import unittest
import argparse

from hmcollab.tests.test_articles import TestArticles
from hmcollab.tests.test_datasets import TestDatasets
from hmcollab.tests.test_directories import TestDirectories
from hmcollab.tests.test_example import TestExample
from hmcollab.tests.test_scoring import TestScoring
from hmcollab.tests.test_transactions import TestTransactions

# integration tests
from hmcollab.tests.integration_tests.test_data_exists import IntegrationTestDataExists


class CountSuite(object):
    def __init__(self):
        self.count = 0
        self.s = unittest.TestSuite()

    def add(self, tests):
        self.count += 1
        print("%d: %s" % (self.count, tests.__name__))
        self.s.addTest(unittest.makeSuite(tests))


def suite(integration):
    s = CountSuite()

    s.add(TestArticles)
    s.add(TestDatasets)
    s.add(TestDirectories)
    s.add(TestExample)
    s.add(TestScoring)
    s.add(TestTransactions)

    if integration:
        print("Running integration tests.")
        s.add(IntegrationTestDataExists)

    return s.s


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tests for HM Collab Project")
    parser.add_argument(
        "--integration",
        help="Whether to call tests which require dataset to be installed and generated files.",
        default=None
    )
    args = parser.parse_args()
    runner = unittest.TextTestRunner()
    runner.run(suite(args.integration))
