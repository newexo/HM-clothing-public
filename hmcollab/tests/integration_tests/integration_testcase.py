import hashlib
from unittest import TestCase


class IntegrationTestCase(TestCase):
    def hash(self, data):
        return hashlib.md5(repr(data).encode("utf8")).hexdigest()
