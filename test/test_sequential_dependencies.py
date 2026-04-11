import unittest
import pytest

@pytest.mark.prototype
class TestSequentialDependencies(unittest.TestCase):
    @pytest.mark.prototype
    def test_sequential_dependencies(self):
        self.fail("Not Implemented")

if __name__ == '__main__':
    unittest.main()
