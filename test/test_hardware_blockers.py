import unittest
import pytest

@pytest.mark.prototype
class TestHardwareBlockers(unittest.TestCase):
    """Prototype tests for hardware blocking boundaries."""
    @pytest.mark.prototype
    def test_hardware_blockers(self):
        self.fail("Not implemented")

if __name__ == '__main__':
    unittest.main()
