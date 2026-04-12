import unittest
import pytest
from tinygrad import Tensor, dtypes
from tinygrad.helpers import Context
import shutil

def has_compiler(): return shutil.which("riscv64-unknown-elf-gcc") is not None

@pytest.mark.prototype
class TestW8A8LateDequantization(unittest.TestCase):
    @pytest.mark.prototype
    @unittest.skipIf(not has_compiler(), "Missing riscv64-unknown-elf-gcc")
    def test_vxsat_assertion(self):
        with Context(DEV="CORALNPU"):
            a = Tensor([127], dtype=dtypes.int8).realize()
            self.assertEqual(a.numpy()[0], 127)

if __name__ == '__main__':
    unittest.main()
