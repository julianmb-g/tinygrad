import unittest
import pytest
from tinygrad import Tensor, dtypes
from tinygrad.helpers import Context
import shutil

def has_compiler(): return shutil.which("riscv64-unknown-elf-gcc") is not None

@pytest.mark.prototype
class TestTunerGradients(unittest.TestCase):
    @pytest.mark.prototype
    @unittest.skipIf(not has_compiler(), "Missing riscv64-unknown-elf-gcc")
    def test_proxy_hazard_weights(self):
        pass

if __name__ == '__main__':
    unittest.main()
