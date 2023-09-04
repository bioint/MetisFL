
import unittest

from metisfl.server.store.hash_map import HashMapModelStore
from metisfl.proto import model_pb2


class TestHashMapModelStore(unittest.TestCase):

    def setUp(self) -> None:
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_insert(self):
        pairs = [
            ("l1", model_pb2.Model()),
            ("l2", model_pb2.Model()),
        ]
        store = HashMapModelStore(0)
        store.insert(pairs)
        self.assertEqual(len(store.store_cache), 2)

    def test_select(self):
        pairs = (
            ("l1", "model_pb2.Model()"),
            ("l2", "model_pb2.Model()"),
            ("l1", "model_pb2.Model()"),
            ("l2", "model_pb2.Model()"),
        )
        store = HashMapModelStore(0)
        store.insert(pairs)
        selected = store.select([["l1", 2]])

        self.assertEqual(len(selected["l1"]), 2)

    def test_capacity(self):
        pairs = (
            ("l1", "model_pb2.Model()"),
            ("l1", "model_pb2.Model()"),
            ("l1", "model_pb2.Model()"),
            ("l1", "model_pb2.Model()"),
        )

        store = HashMapModelStore(2)
        store.insert(pairs)
        self.assertEqual(len(store.store_cache["l1"]), 2)


if __name__ == '__main__':
    unittest.main()
