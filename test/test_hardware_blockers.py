import pytest
import unittest

@pytest.mark.prototype
class TestHardwareBlockers(unittest.TestCase):
    """Prototype tests for hardware blocking boundaries."""
    @pytest.mark.prototype
    def test_hardware_blockers(self):
        pass

@pytest.mark.prototype
def test_hardware_blocker_2():
    pytest.fail("Not Implemented")
