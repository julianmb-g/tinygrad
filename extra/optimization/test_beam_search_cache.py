import unittest
from tinygrad.codegen.opt.search import UOpAstHash, UOpMemoizationCache

class TestUOpMemoizationCache(unittest.TestCase):
    def test_uop_ast_hash_equality(self):
        h1 = UOpAstHash(1, 2, 3, 4)
        h2 = UOpAstHash(1, 2, 3, 4)
        h3 = UOpAstHash(1, 2, 3, 5)
        
        self.assertEqual(h1, h2)
        self.assertNotEqual(h1, h3)
        self.assertEqual(hash(h1), hash(h2))
        self.assertNotEqual(hash(h1), hash(h3))

    def test_uop_memoization_cache_lru(self):
        cache = UOpMemoizationCache(max_capacity=3)
        h1 = UOpAstHash(1, 0, 0, 0)
        h2 = UOpAstHash(2, 0, 0, 0)
        h3 = UOpAstHash(3, 0, 0, 0)
        h4 = UOpAstHash(4, 0, 0, 0)

        cache.insert(h1, 10.0)
        cache.insert(h2, 20.0)
        cache.insert(h3, 30.0)
        
        self.assertEqual(cache.get_empirical_cycles(h1), 10.0)
        self.assertEqual(cache.get_empirical_cycles(h2), 20.0)
        self.assertEqual(cache.get_empirical_cycles(h3), 30.0)

        cache.get_empirical_cycles(h1)
        cache.get_empirical_cycles(h2)
        cache.insert(h4, 40.0)

        # h3 should be evicted
        self.assertIsNone(cache.get_empirical_cycles(h3))
        self.assertEqual(cache.get_empirical_cycles(h1), 10.0)
        self.assertEqual(cache.get_empirical_cycles(h4), 40.0)

if __name__ == '__main__':
    unittest.main()
